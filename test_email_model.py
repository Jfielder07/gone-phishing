"""
Tests for the Email class. Run from the project folder with:
    python -m unittest test_email_model -v
"""
import unittest
from pathlib import Path

from email_model import Email

HERE = Path(__file__).parent


class TestEmail(unittest.TestCase):
    def test_plain_text_email(self):
        raw = (
            b"From: Prof Smith <smith@university.edu>\r\n"
            b"To: josh@example.com\r\n"
            b"Subject: Meeting moved\r\n\r\n"
            b"See the agenda at https://university.edu/agenda.\r\n"
        )
        email = Email("meeting.eml", raw)
        self.assertEqual(email.filename, "meeting.eml")
        self.assertIsNone(email.email_id)
        self.assertEqual(email.get_sender(), "Prof Smith <smith@university.edu>")
        self.assertEqual(email.subject, "Meeting moved")
        self.assertIn("See the agenda", email.get_body())
        # trailing period removed from the link
        self.assertEqual(email.links, ["https://university.edu/agenda"])

    def test_phishing_sample(self):
        email = Email("sample_phishing.eml", (HERE / "sample_phishing.eml").read_bytes())
        self.assertEqual(email.get_sender(), "PayPal Security <security@paypa1-alerts.com>")
        # encoded subject header is decoded
        self.assertIn("Urgent: Your account has been suspended", email.subject)
        # body uses the plain-text part, not raw HTML
        self.assertIn("unusual activity", email.get_body())
        self.assertNotIn("<p>", email.get_body())
        # the hidden href is caught, not just the address shown on screen
        self.assertIn("http://paypa1-alerts.com/login.php?id=8812", email.links)
        self.assertIn("https://www.paypal.com/verify", email.links)
        self.assertEqual(len(email.links), 2)  # duplicates removed

    def test_html_only_email(self):
        raw = (
            b"From: a@b.com\r\nSubject: hi\r\nContent-Type: text/html\r\n\r\n"
            b"<html><style>p{color:red}</style><body><p>Hello <b>there</b></p>"
            b"<a href='https://x.com/a'>click</a> <a href='mailto:a@b.com'>mail</a></body></html>"
        )
        email = Email("h.eml", raw)
        self.assertEqual(email.get_body(), "Hello there\nclick mail")
        self.assertEqual(email.links, ["https://x.com/a"])  # mailto skipped

    def test_missing_subject_and_no_links(self):
        email = Email("x.eml", b"From: a@b.com\r\nTo: c@d.com\r\n\r\nno links here")
        self.assertEqual(email.subject, "")
        self.assertEqual(email.links, [])

    def test_bad_charset_does_not_crash(self):
        raw = b"From: a@b.com\r\nTo: c@d.com\r\nContent-Type: text/plain; charset=not-real\r\n\r\nhello"
        self.assertIn("hello", Email("x.eml", raw).get_body())


if __name__ == "__main__":
    unittest.main()
