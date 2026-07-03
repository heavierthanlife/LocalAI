"""Simple SMTP email utility for system notifications."""
import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from threading import Thread

logger = logging.getLogger(__name__)

SMTP_CONFIG = {
    'host': os.getenv('SMTP_HOST', ''),
    'port': int(os.getenv('SMTP_PORT', '587')),
    'user': os.getenv('SMTP_USER', ''),
    'password': os.getenv('SMTP_PASSWORD', ''),
    'from_addr': os.getenv('SMTP_FROM', 'noreply@ai-services.local'),
    'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
}


def is_configured():
    """Check if SMTP is configured."""
    return bool(SMTP_CONFIG['host'] and SMTP_CONFIG['user'] and SMTP_CONFIG['password'])


def send_email(to_addr, subject, body, html_body=None, async_mode=True):
    """Send an email. Runs in background thread if async_mode=True."""
    if not is_configured():
        logger.warning(f"Email not sent (SMTP not configured): {subject}")
        return False

    def _send():
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = SMTP_CONFIG['from_addr']
            msg['To'] = to_addr
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            if html_body:
                msg.attach(MIMEText(html_body, 'html', 'utf-8'))

            if SMTP_CONFIG['use_tls']:
                server = smtplib.SMTP(SMTP_CONFIG['host'], SMTP_CONFIG['port'], timeout=10)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(SMTP_CONFIG['host'], SMTP_CONFIG['port'], timeout=10)

            server.login(SMTP_CONFIG['user'], SMTP_CONFIG['password'])
            server.sendmail(SMTP_CONFIG['from_addr'], to_addr, msg.as_string())
            server.quit()
            logger.info(f"Email sent to {to_addr}: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email to {to_addr}: {e}")

    if async_mode:
        Thread(target=_send, daemon=True, name="email-sender").start()
    else:
        _send()
    return True


def notify_admin(subject, body):
    """Send notification to the admin email."""
    admin_email = os.getenv('ADMIN_EMAIL', '')
    if admin_email:
        send_email(admin_email, f"[AI_Services] {subject}", body)


def notify_new_user(username):
    """Notify admin of new user registration."""
    notify_admin(
        "New User Registered",
        f"User '{username}' has created an account."
    )


def notify_error(error_msg):
    """Notify admin of critical errors."""
    notify_admin("System Error", f"Error: {error_msg}")
