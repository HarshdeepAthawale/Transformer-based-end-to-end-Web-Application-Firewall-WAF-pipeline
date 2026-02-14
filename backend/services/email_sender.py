"""
Email sender for WAF alert notifications.
Uses SMTP from config; no-op if SMTP is not configured.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List
from loguru import logger

from backend.config import config


def is_configured() -> bool:
    """Return True if SMTP is configured for sending."""
    return bool(config.SMTP_HOST)


def send_alert_email(
    to_emails: List[str],
    subject: str,
    body_text: str,
    body_html: str = "",
) -> bool:
    """
    Send an alert email to the given recipients.
    Returns True if sent successfully, False otherwise.
    No-op if SMTP is not configured.
    """
    if not to_emails:
        logger.debug("No alert email recipients configured")
        return False
    if not is_configured():
        logger.debug("SMTP not configured; skipping alert email")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = config.ALERT_FROM_EMAIL
        msg["To"] = ", ".join(to_emails)

        msg.attach(MIMEText(body_text, "plain"))
        if body_html:
            msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as smtp:
            if config.SMTP_USE_TLS:
                smtp.starttls()
            if config.SMTP_USER and config.SMTP_PASSWORD:
                smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
            smtp.sendmail(config.ALERT_FROM_EMAIL, to_emails, msg.as_string())

        logger.info(f"Alert email sent to {len(to_emails)} recipient(s)")
        return True
    except Exception as e:
        logger.warning(f"Failed to send alert email: {e}")
        return False
