"""
Email
-----
Data object representing one uploaded email.

Takes the raw bytes of a validated .eml file and pulls out the pieces the rest
of the system needs: who sent it, the subject, the readable body text, and
every link it contains.

Usage:
    email = Email("invoice.eml", raw_bytes)
    email.get_sender()   # 'PayPal Support <support@paypa1-security.com>'
    email.links          # ['http://paypa1-security.com/verify']
"""
import re
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser

# Matches http:// or https:// URLs, plus bare www. addresses, in plain text
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
# Punctuation that often sits right after a URL in a sentence but isn't part of it
TRAILING_PUNCTUATION = ".,;:!?)]}'\""


class _HTMLTextAndLinks(HTMLParser):
    """Helper: reads an HTML email body, collecting its visible text and the href of every <a> tag."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.links = []
        self._skip_depth = 0  # >0 while inside <script> or <style>, whose contents aren't visible text

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip_depth += 1
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href.strip())
        elif tag in ("br", "p", "div", "tr", "li"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1
        elif tag in ("p", "div", "tr", "li"):
            self.text_parts.append("\n")  # so text from separate blocks doesn't run together

    def handle_data(self, data):
        if not self._skip_depth:
            self.text_parts.append(data)

    def get_text(self):
        text = "".join(self.text_parts)
        lines = (" ".join(line.split()) for line in text.splitlines())  # collapse extra spaces
        return "\n".join(line for line in lines if line)  # drop blank lines


class Email:
    def __init__(self, filename, raw_content, email_id=None):
        """
        filename:    the uploaded file's name, e.g. 'invoice.eml'
        raw_content: the file's bytes (already checked by EmailValidator)
        email_id:    the database ID; None until DatabaseManager saves the email
        """
        self.email_id = email_id
        self.filename = filename
        self.sender = ""
        self.subject = ""
        self.body = ""
        self.links = []

        self._raw_content = raw_content
        self.extract_content()

    def extract_content(self):
        """Parse the raw .eml bytes and fill in sender, subject, body and links."""
        message = BytesParser(policy=policy.default).parsebytes(self._raw_content)

        # policy.default decodes encoded headers, e.g. '=?utf-8?b?...?=' becomes readable text
        self.sender = str(message.get("From", "")).strip()
        self.subject = str(message.get("Subject", "")).strip()

        plain_text = self._get_part_text(message, "plain")
        html_text = self._get_part_text(message, "html")

        html_links = []
        if html_text:
            html_reader = _HTMLTextAndLinks()
            html_reader.feed(html_text)
            html_links = html_reader.links

        # Prefer the plain-text version for the body; fall back to text pulled out of the HTML
        if plain_text:
            self.body = plain_text.strip()
        elif html_text:
            self.body = html_reader.get_text()
        else:
            self.body = ""

        # Links come from <a href="..."> tags AND any URLs written out in the text.
        # The href matters most: phishing emails often show one address but link to another.
        found = html_links + URL_PATTERN.findall(plain_text) + URL_PATTERN.findall(html_text)
        self.links = self._clean_links(found)

    def get_sender(self):
        """Return the From header, e.g. 'PayPal Support <support@paypa1-security.com>'."""
        return self.sender

    def get_body(self):
        """Return the readable text of the email."""
        return self.body

    # ---- private helpers ----
    @staticmethod
    def _get_part_text(message, subtype):
        """Return the decoded text of the email's text/<subtype> part, or '' if there isn't one."""
        part = message.get_body(preferencelist=(subtype,))
        if part is None:
            return ""
        try:
            return part.get_content()
        except (LookupError, UnicodeError):
            # Unknown or wrong character set: decode the raw bytes as best we can
            payload = part.get_payload(decode=True) or b""
            return payload.decode("utf-8", errors="replace")

    @staticmethod
    def _clean_links(links):
        """Strip trailing punctuation, keep only web links, and remove duplicates (keeping order)."""
        cleaned = []
        for link in links:
            link = link.strip().rstrip(TRAILING_PUNCTUATION)
            if not link.lower().startswith(("http://", "https://", "www.")):
                continue  # skip mailto:, tel:, '#' anchors, etc.
            if link not in cleaned:
                cleaned.append(link)
        return cleaned

    def __repr__(self):
        return f"Email(id={self.email_id}, filename={self.filename!r}, sender={self.sender!r}, links={len(self.links)})"
