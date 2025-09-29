// frontend/src/components/forms/DataUploadForm.tsx

"use client";

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, File as FileIcon, X, CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { useDataUpload } from '@/hooks';

interface UploadableFile {
  file: File;
  errors: string[];
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'error';
}

interface DataUploadFormProps {
  onUpload?: (files: any) => void;
  onUploadComplete?: (results: any) => void;
  acceptedTypes?: string[];
  acceptedFileTypes?: Record<string, string[]>;
  maxFileSize?: number;
  title?: string;
  description?: string;
}

export const DataUploadForm: React.FC<DataUploadFormProps> = ({
  onUpload,
  onUploadComplete,
  acceptedTypes,
  acceptedFileTypes = {
    'text/csv': ['.csv'],
    'application/json': ['.json'],
    'text/plain': ['.txt']
  },
  maxFileSize = 10 * 1024 * 1024, // 10MB
}) => {
  const [files, setFiles] = useState<UploadableFile[]>([]);
  const { uploadFile, isUploading } = useDataUpload();

  const validateFile = useCallback((file: File): string[] => {
    const errors: string[] = [];

    if (file.size > maxFileSize) {
      errors.push(`File size exceeds ${Math.round(maxFileSize / 1024 / 1024)}MB limit`);
    }

    const allowedTypes = Object.keys(acceptedFileTypes);
    if (!allowedTypes.includes(file.type)) {
      errors.push('File type not supported');
    }

    return errors;
  }, [acceptedFileTypes, maxFileSize]);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    const newFiles = acceptedFiles.map(file => ({
      file,
      errors: validateFile(file),
      progress: 0,
      status: 'pending' as const,
    }));

    // Handle rejected files
    const rejectedUploadFiles = rejectedFiles.map(({ file, errors }) => ({
      file,
      errors: errors.map((e: any) => e.message),
      progress: 0,
      status: 'error' as const,
    }));

    setFiles(prev => [...prev, ...newFiles, ...rejectedUploadFiles]);
  }, [validateFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFileTypes,
    maxSize: maxFileSize,
    multiple: true,
  });

  const removeFile = (fileName: string) => {
    setFiles(prev => prev.filter(f => f.file.name !== fileName));
  };

  const handleUpload = async () => {
    const validFiles = files.filter(f => f.errors.length === 0 && f.status === 'pending');

    for (const uploadableFile of validFiles) {
      try {
        // Update status to uploading
        setFiles(prev => prev.map(f =>
          f.file.name === uploadableFile.file.name
            ? { ...f, status: 'uploading' as const }
            : f
        ));

        // Simulate progress updates
        const progressInterval = setInterval(() => {
          setFiles(prev => prev.map(f =>
            f.file.name === uploadableFile.file.name && f.progress < 90
              ? { ...f, progress: f.progress + 10 }
              : f
          ));
        }, 200);

        const result = await uploadFile(uploadableFile.file);

        clearInterval(progressInterval);

        // Update to completed
        setFiles(prev => prev.map(f =>
          f.file.name === uploadableFile.file.name
            ? { ...f, status: 'completed' as const, progress: 100 }
            : f
        ));

        if (onUploadComplete) {
          onUploadComplete(result);
        }
      } catch (error) {
        setFiles(prev => prev.map(f =>
          f.file.name === uploadableFile.file.name
            ? {
                ...f,
                status: 'error' as const,
                errors: [...f.errors, error instanceof Error ? error.message : 'Upload failed']
              }
            : f
        ));
      }
    }
  };

  const validFilesCount = files.filter(f => f.errors.length === 0).length;
  const completedFilesCount = files.filter(f => f.status === 'completed').length;

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="rounded-lg border bg-white p-8 shadow-sm dark:border-slate-800 dark:bg-slate-950">
        <div className="mb-6">
          <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-50 mb-2">
            Upload Dataset
          </h2>
          <p className="text-slate-600 dark:text-slate-400">
            Upload CSV, JSON, or TXT files for analysis. Maximum file size: {Math.round(maxFileSize / 1024 / 1024)}MB
          </p>
        </div>

        {/* Drag and Drop Area */}
        <div
          {...getRootProps()}
          className={`relative p-10 border-2 border-dashed rounded-lg cursor-pointer transition-all duration-200 ease-in-out ${
            isDragActive
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/20'
              : 'border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/50 dark:border-slate-700 dark:bg-slate-800/50 dark:hover:border-blue-500'
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center justify-center text-center">
            <UploadCloud className="w-12 h-12 text-slate-400 dark:text-slate-500 mb-4" />
            <p className="text-lg font-medium text-slate-600 dark:text-slate-300">
              {isDragActive ? "Drop the files here..." : "Drag & drop files here, or click to select"}
            </p>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              Supports: {Object.values(acceptedFileTypes).flat().join(', ')}
            </p>
          </div>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="mt-8 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-700 dark:text-slate-300">
                Selected Files ({completedFilesCount}/{validFilesCount} completed)
              </h3>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setFiles([])}
                disabled={isUploading}
              >
                Clear All
              </Button>
            </div>

            <div className="space-y-3">
              {files.map((uploadableFile, index) => (
                <div key={index} className="flex items-center bg-slate-50 dark:bg-slate-800/50 p-4 rounded-lg border border-slate-200 dark:border-slate-700">
                  <FileIcon className="w-6 h-6 text-slate-500 dark:text-slate-400" />

                  <div className="ml-4 flex-grow">
                    <div className="flex items-center gap-2 mb-1">
                      <p className="font-medium text-slate-800 dark:text-slate-200">
                        {uploadableFile.file.name}
                      </p>
                      {uploadableFile.status === 'completed' && (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      )}
                      {uploadableFile.status === 'error' && (
                        <AlertCircle className="w-4 h-4 text-red-500" />
                      )}
                      {uploadableFile.errors.length === 0 && uploadableFile.status === 'pending' && (
                        <Badge variant="secondary">Ready</Badge>
                      )}
                    </div>

                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      {(uploadableFile.file.size / 1024).toFixed(2)} KB
                    </p>

                    {uploadableFile.status === 'uploading' && (
                      <div className="mt-2">
                        <Progress value={uploadableFile.progress} className="h-2" />
                      </div>
                    )}

                    {uploadableFile.errors.length > 0 && (
                      <div className="mt-2">
                        {uploadableFile.errors.map((error, errorIndex) => (
                          <p key={errorIndex} className="text-sm text-red-600 dark:text-red-400">
                            {error}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>

                  <button
                    onClick={() => removeFile(uploadableFile.file.name)}
                    disabled={isUploading}
                    className="p-1 text-slate-400 hover:text-red-500 rounded-full transition-colors disabled:opacity-50"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="mt-8 flex justify-end gap-3">
          {files.length > 0 && (
            <Button
              variant="outline"
              onClick={() => setFiles([])}
              disabled={isUploading}
            >
              Clear All
            </Button>
          )}
          <Button
            size="lg"
            onClick={handleUpload}
            disabled={validFilesCount === 0 || isUploading}
            className="min-w-[120px]"
          >
            {isUploading ? 'Uploading...' : `Upload ${validFilesCount} ${validFilesCount === 1 ? 'File' : 'Files'}`}
          </Button>
        </div>
      </div>
    </div>
  );
};