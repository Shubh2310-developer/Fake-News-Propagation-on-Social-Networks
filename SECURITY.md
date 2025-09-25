# Security Policy

## Overview

The Fake News Game Theory project takes security seriously. As a research platform handling sensitive data and providing machine learning services, we are committed to maintaining the highest standards of security for our users, contributors, and the research community.

## Supported Versions

We actively maintain security updates for the following versions:

| Version | Supported          | Support Level |
| ------- | ------------------ | ------------- |
| 1.x.x   |  Full support    | Security patches, bug fixes, feature updates |
| 0.2.x   |  Limited support | Critical security patches only |
| 0.1.x   | L End of life     | No security updates |

## Reporting a Vulnerability

### =¨ For Security Issues - DO NOT Create Public Issues

If you discover a security vulnerability, please **DO NOT** create a public GitHub issue. Instead, report it privately using one of the methods below:

### Primary Reporting Channel

=ç **Email**: [security@fake-news-game-theory.org](mailto:security@fake-news-game-theory.org)

**Response Time**: We aim to acknowledge security reports within 48 hours and provide regular updates every 72 hours until resolution.

### What to Include in Your Report

Please provide as much of the following information as possible:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential impact and attack scenarios
3. **Reproduction Steps**: Detailed steps to reproduce the issue
4. **Proof of Concept**: If applicable, include PoC code or screenshots
5. **Affected Versions**: Which versions are affected
6. **Suggested Fix**: If you have ideas for mitigation or patches
7. **Your Details**: Name and contact info for follow-up (optional)

### Example Security Report Template

```
Subject: [SECURITY] Vulnerability in ML Model API

**Vulnerability Type**: [e.g., SQL Injection, XSS, Authentication Bypass]

**Affected Component**: [e.g., Classifier API, Frontend Dashboard, Database]

**Severity**: [Low/Medium/High/Critical]

**Description**:
[Detailed description of the vulnerability]

**Steps to Reproduce**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Impact**:
[What an attacker could accomplish with this vulnerability]

**Proof of Concept**:
[Code, screenshots, or other evidence]

**Suggested Mitigation**:
[Your thoughts on how to fix this]
```

## Security Response Process

### Our Commitment

When you report a security vulnerability:

1. **Acknowledgment**: We'll acknowledge receipt within 48 hours
2. **Investigation**: We'll investigate and validate the report within 7 days
3. **Resolution**: We'll work on a fix with appropriate priority
4. **Communication**: We'll keep you updated on progress
5. **Credit**: We'll credit you in our security advisories (if desired)
6. **Disclosure**: We'll coordinate responsible public disclosure

### Response Timeline

- **Critical Vulnerabilities**: Patches within 24-72 hours
- **High Severity**: Patches within 7 days
- **Medium Severity**: Patches within 30 days
- **Low Severity**: Patches in next regular release

## Security Measures in Place

### Backend Security (FastAPI)

- **Authentication**: JWT-based authentication with secure tokens
- **Authorization**: Role-based access control (RBAC)
- **Input Validation**: Comprehensive input sanitization using Pydantic
- **Rate Limiting**: API rate limiting to prevent abuse
- **CORS**: Properly configured Cross-Origin Resource Sharing
- **HTTPS Only**: All production traffic uses TLS/SSL encryption
- **Database Security**: Parameterized queries to prevent SQL injection
- **Secrets Management**: Environment variables for sensitive configuration

### Frontend Security (Next.js)

- **Content Security Policy (CSP)**: Strict CSP headers
- **XSS Protection**: Built-in React XSS protections
- **CSRF Protection**: CSRF tokens for state-changing operations
- **Secure Headers**: Security headers via Next.js middleware
- **Client-Side Validation**: Input validation on frontend
- **Secure Storage**: Secure token storage practices
- **Dependency Scanning**: Regular npm audit checks

### Infrastructure Security

- **Container Security**: Minimal Docker images with non-root users
- **Network Security**: Private networks and firewall rules
- **Secrets Management**: Docker secrets and environment variables
- **Regular Updates**: Automated security updates for base images
- **Access Control**: Limited access to production systems
- **Monitoring**: Security monitoring and alerting
- **Backup Security**: Encrypted backups with access controls

### Data Protection

- **Data Encryption**: At-rest and in-transit encryption
- **Privacy by Design**: Minimal data collection and retention
- **Anonymization**: User data anonymization where possible
- **Access Logging**: Comprehensive audit logs
- **Data Retention**: Clear data retention and deletion policies
- **Compliance**: GDPR and research data compliance measures

## Vulnerability Disclosure Policy

### Coordinated Disclosure

We follow a **coordinated disclosure** policy:

1. **Private Disclosure**: Report sent privately to our security team
2. **Investigation Period**: 90 days maximum for investigation and patching
3. **Public Disclosure**: Joint announcement after patch is available
4. **Credit**: Public acknowledgment of the researcher (if desired)

### Hall of Fame

We maintain a security researchers hall of fame to recognize contributors:

- Researchers who report valid security vulnerabilities
- Contributors who help improve our security posture
- Community members who promote secure practices

## Security Best Practices for Contributors

### Code Security Guidelines

- **Input Validation**: Always validate and sanitize user inputs
- **Authentication**: Never bypass authentication checks
- **Authorization**: Implement proper access controls
- **Error Handling**: Don't expose sensitive information in errors
- **Logging**: Log security events but avoid logging sensitive data
- **Dependencies**: Use only trusted dependencies and keep them updated
- **Secrets**: Never commit secrets, keys, or passwords to version control

### Secure Development Process

- **Code Review**: All security-sensitive code requires review
- **Static Analysis**: Automated security scanning in CI/CD
- **Dynamic Testing**: Regular penetration testing
- **Dependency Audits**: Regular checks for vulnerable dependencies
- **Security Training**: Ongoing security awareness for contributors

## Threat Model

### Assets We Protect

1. **User Data**: Personal information, research data, usage patterns
2. **ML Models**: Trained models, training data, model parameters
3. **Research Data**: Datasets, experimental results, analysis outputs
4. **System Integrity**: Application logic, database integrity, service availability
5. **Intellectual Property**: Research methodologies, algorithmic innovations

### Primary Threat Actors

- **Malicious Users**: Attempting to manipulate research results
- **Data Thieves**: Seeking to steal sensitive research data
- **System Attackers**: Trying to compromise infrastructure
- **Insider Threats**: Malicious or negligent insiders
- **Nation-State Actors**: Advanced persistent threats targeting research

### Attack Vectors

- Web application vulnerabilities (OWASP Top 10)
- API security issues
- Social engineering attacks
- Supply chain attacks on dependencies
- Infrastructure misconfigurations
- Data poisoning attacks on ML models

## Security Contact Information

### Primary Security Team

- **Security Lead**: security-lead@fake-news-game-theory.org
- **Technical Lead**: tech-security@fake-news-game-theory.org
- **Research Ethics**: ethics@fake-news-game-theory.org

### Emergency Contact

For critical security incidents requiring immediate attention:

=ñ **Emergency Hotline**: [+1-XXX-XXX-XXXX] (24/7)

### PGP Key

For encrypted communications:

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
[PGP Public Key Block would go here]
-----END PGP PUBLIC KEY BLOCK-----
```

**Key ID**: [Key ID]
**Fingerprint**: [Full fingerprint]

## Security Updates and Advisories

### Where to Find Security Information

- **GitHub Security Advisories**: https://github.com/fake-news-game-theory/security/advisories
- **Security Mailing List**: security-announcements@fake-news-game-theory.org
- **RSS Feed**: https://fake-news-game-theory.org/security/feed.xml
- **Twitter**: @FakeNewsGameTheory (for urgent announcements)

### Notification Preferences

Subscribe to our security mailing list for:
- Security advisories
- Patch notifications
- Security best practices updates
- Incident reports (post-mortem)

## Legal and Responsible Disclosure

### Safe Harbor

We commit to:

- Not pursue legal action against researchers who:
  - Act in good faith
  - Follow responsible disclosure practices
  - Do not violate user privacy
  - Do not disrupt our services
  - Do not access data beyond what's necessary to demonstrate the vulnerability

### Scope

This policy covers:
-  **In Scope**: Main application, API endpoints, admin interfaces, infrastructure
- L **Out of Scope**: Third-party services, social engineering, physical attacks

### Research Guidelines

When conducting security research:
- **Minimize Impact**: Don't disrupt services or access user data
- **Respect Privacy**: Don't access or modify user data
- **Document Carefully**: Maintain detailed records of your testing
- **Report Promptly**: Report vulnerabilities as soon as identified

## Incident Response

### In Case of a Security Incident

If we detect a security incident:

1. **Immediate Response**: Contain the incident and assess impact
2. **User Notification**: Notify affected users within 72 hours
3. **Public Communication**: Transparent communication about the incident
4. **Post-Incident Review**: Conduct thorough post-mortem analysis
5. **Improvements**: Implement measures to prevent similar incidents

### User Actions During Incidents

If you suspect a security incident:
1. **Change Passwords**: Update your account passwords
2. **Review Activity**: Check your account for unauthorized access
3. **Report Suspicious Activity**: Contact our security team
4. **Monitor Communications**: Watch for official updates

## Acknowledgments

We thank the security research community for helping us maintain the security of our platform. Special recognition to:

- [Security Researcher Names] - Responsible disclosure of vulnerabilities
- [Bug Bounty Participants] - Ongoing security testing
- [Security Advisors] - Strategic security guidance

---

**Remember**: Security is everyone's responsibility. If you see something suspicious, report it. Together, we can maintain a secure research environment for combating misinformation.