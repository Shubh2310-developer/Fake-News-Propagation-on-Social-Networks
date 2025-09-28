#!/bin/bash

# User data script for EKS worker nodes
# This script is executed when EC2 instances are launched as part of EKS node groups

set -o xtrace

# Set up logging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

# Update the system
yum update -y

# Install additional packages
yum install -y \
    aws-cli \
    amazon-cloudwatch-agent \
    htop \
    iotop \
    jq \
    unzip \
    wget

# Configure CloudWatch agent
cat <<EOF > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
{
    "agent": {
        "metrics_collection_interval": 60,
        "run_as_user": "root"
    },
    "metrics": {
        "namespace": "CWAgent",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "netstat": {
                "measurement": [
                    "tcp_established",
                    "tcp_time_wait"
                ],
                "metrics_collection_interval": 60
            },
            "swap": {
                "measurement": [
                    "swap_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/messages",
                        "log_group_name": "/aws/eks/${cluster_name}/worker-logs",
                        "log_stream_name": "{instance_id}/messages"
                    },
                    {
                        "file_path": "/var/log/dmesg",
                        "log_group_name": "/aws/eks/${cluster_name}/worker-logs",
                        "log_stream_name": "{instance_id}/dmesg"
                    }
                ]
            }
        }
    }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s

# Bootstrap the node to join the EKS cluster
/etc/eks/bootstrap.sh ${cluster_name} ${bootstrap_arguments}

# Configure kubelet for better resource management
cat <<EOF > /etc/kubernetes/kubelet/kubelet-config.json
{
    "kind": "KubeletConfiguration",
    "apiVersion": "kubelet.config.k8s.io/v1beta1",
    "address": "0.0.0.0",
    "port": 10250,
    "readOnlyPort": 0,
    "cgroupDriver": "systemd",
    "hairpinMode": "hairpin-veth",
    "serializeImagePulls": false,
    "featureGates": {
        "RotateKubeletServerCertificate": true
    },
    "clusterDomain": "cluster.local",
    "clusterDNS": ["10.100.0.10"],
    "streamingConnectionIdleTimeout": "4h0m0s",
    "nodeStatusUpdateFrequency": "10s",
    "imageMinimumGCAge": "2m0s",
    "imageGCHighThresholdPercent": 85,
    "imageGCLowThresholdPercent": 80,
    "evictionHard": {
        "memory.available": "100Mi",
        "nodefs.available": "10%",
        "nodefs.inodesFree": "5%"
    },
    "evictionSoft": {
        "memory.available": "200Mi",
        "nodefs.available": "15%",
        "nodefs.inodesFree": "10%"
    },
    "evictionSoftGracePeriod": {
        "memory.available": "2m0s",
        "nodefs.available": "2m0s",
        "nodefs.inodesFree": "2m0s"
    },
    "maxPods": 110,
    "kubeReserved": {
        "cpu": "100m",
        "memory": "100Mi",
        "ephemeral-storage": "1Gi"
    },
    "systemReserved": {
        "cpu": "100m",
        "memory": "100Mi",
        "ephemeral-storage": "1Gi"
    }
}
EOF

# Restart kubelet to apply configuration
systemctl daemon-reload
systemctl restart kubelet

# Install and configure node exporter for Prometheus monitoring
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v1.6.0/node_exporter-1.6.0.linux-amd64.tar.gz
tar xvfz node_exporter-1.6.0.linux-amd64.tar.gz
cp node_exporter-1.6.0.linux-amd64/node_exporter /usr/local/bin/
useradd --no-create-home --shell /bin/false node_exporter
chown node_exporter:node_exporter /usr/local/bin/node_exporter

# Create systemd service for node exporter
cat <<EOF > /etc/systemd/system/node_exporter.service
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter --web.listen-address=:9100

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable node_exporter
systemctl start node_exporter

# Configure log rotation for container logs
cat <<EOF > /etc/logrotate.d/docker-container
/var/log/containers/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 0644 root root
    postrotate
        docker kill -s USR1 \$(docker ps -q) 2>/dev/null || true
    endscript
}
EOF

# Set up disk usage monitoring
cat <<EOF > /usr/local/bin/disk-usage-alert.sh
#!/bin/bash
THRESHOLD=80
USAGE=\$(df / | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{ print \$5 }' | sed 's/%//g')

if [ \$USAGE -gt \$THRESHOLD ]; then
    echo "Disk usage is above \$THRESHOLD%. Current usage: \$USAGE%"
    # Log to CloudWatch
    aws logs put-log-events \
        --log-group-name "/aws/eks/${cluster_name}/worker-logs" \
        --log-stream-name "\$(curl -s http://169.254.169.254/latest/meta-data/instance-id)/disk-alerts" \
        --log-events timestamp=\$(date +%s000),message="Disk usage alert: \$USAGE%"
fi
EOF

chmod +x /usr/local/bin/disk-usage-alert.sh

# Set up cron job for disk monitoring
echo "*/5 * * * * /usr/local/bin/disk-usage-alert.sh" | crontab -

# Configure memory and swap settings for better container performance
echo 'vm.swappiness=1' >> /etc/sysctl.conf
echo 'vm.max_map_count=262144' >> /etc/sysctl.conf
echo 'net.core.somaxconn=65535' >> /etc/sysctl.conf
sysctl -p

# Install AWS Systems Manager agent if not already present
if ! systemctl is-active --quiet amazon-ssm-agent; then
    yum install -y amazon-ssm-agent
    systemctl enable amazon-ssm-agent
    systemctl start amazon-ssm-agent
fi

# Final log entry
echo "User data script completed successfully at \$(date)" >> /var/log/user-data.log

# Signal successful completion to CloudFormation (if needed)
/opt/aws/bin/cfn-signal -e \$? --stack ${cluster_name} --resource AutoScalingGroup --region \$(curl -s http://169.254.169.254/latest/meta-data/placement/region) || echo "cfn-signal not available"