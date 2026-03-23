# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

**Email:** goering.jared@gmail.com

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

I'll respond within 48 hours and work with you on a fix before any public disclosure.

## Scope

Supermemory stores potentially sensitive data (memories extracted from conversations). Security concerns include:

- **Data leakage** through API endpoints
- **SQL injection** in search or query parameters
- **Unauthorized access** to the memory database
- **Sensitive data in logs** or error messages

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Design Principles

- **Local-first:** Data stays on your machine by default
- **No telemetry:** Nothing phones home
- **API keys stay local:** Only used for LLM calls you configure
