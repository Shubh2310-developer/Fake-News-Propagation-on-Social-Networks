import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * A utility function to conditionally join class names together.
 * It intelligently merges Tailwind CSS classes without style conflicts.
 *
 * @param {...ClassValue[]} inputs - Class names to be combined.
 * @returns {string} The merged class name string.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Formats a number as a currency string.
 * @param {number} amount - The amount to format.
 * @returns {string} The formatted currency string.
 */
export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
}

/**
 * Formats a number with thousands separators.
 * @param {number} num - The number to format.
 * @returns {string} The formatted number string.
 */
export function formatNumber(num: number): string {
  return new Intl.NumberFormat('en-US').format(num);
}

/**
 * Truncates a string to a specified length and adds ellipsis.
 * @param {string} str - The string to truncate.
 * @param {number} length - The maximum length.
 * @returns {string} The truncated string.
 */
export function truncateString(str: string, length: number): string {
  if (str.length <= length) return str;
  return str.slice(0, length) + '...';
}

/**
 * Capitalizes the first letter of a string.
 * @param {string} str - The string to capitalize.
 * @returns {string} The capitalized string.
 */
export function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}