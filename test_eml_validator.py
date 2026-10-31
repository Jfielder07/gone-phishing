"""
Tests for EmailValidator. Run from the project folder with:
    python -m unittest test_eml_validator -v
"""
import unittest

from eml_validator import EmailValidationError, EmailValidator

VALID_EMAIL = (
    b"From: PayPal Support <support@paypa1-security.com>\r\n"
    b"To: josh@example.com\r\n"
    b"Subject: Your account has been suspended\r\n"
    b"Date: Fri, 30 Oct 2026 10:00:00 -0500\r\n"
    b"\r\n"
    b"Click here to verify your account.\r\n"
)


class TestEmailValidator(unittest.TestCase):
    def setUp(self):
        self.validator = EmailValidator()

    def assert_rejected(self, filename, content, expected_message):
        with self.assertRaises(EmailValidationError) as ctx:
            self.validator.validate_file(filename, content)
        self.assertEqual(str(ctx.exception), expected_message)

    # --- validate_file ---
    def test_valid_email_passes(self):
        self.assertTrue(self.validator.validate_file("message.eml", VALID_EMAIL))

    def test_no_file(self):
        self.assert_rejected(None, b"", EmailValidator.NO_FILE_MESSAGE)
        self.assert_rejected("", b"", EmailValidator.NO_FILE_MESSAGE)

    def test_wrong_extension(self):
        self.assert_rejected("message.txt", VALID_EMAIL, EmailValidator.WRONG_TYPE_MESSAGE)

    def test_empty_file(self):
        self.assert_rejected("message.eml", b"", EmailValidator.EMPTY_FILE_MESSAGE)

    def test_too_large(self):
        big = VALID_EMAIL + b"x" * self.validator.max_file_size
        self.assert_rejected("message.eml", big, EmailValidator.TOO_LARGE_MESSAGE)

    def test_not_an_email(self):
        self.assert_rejected("notes.eml", b"just some notes\nno headers here", EmailValidator.BAD_FORMAT_MESSAGE)

    # --- check_extension ---
    def test_extension_is_case_insensitive(self):
        self.assertTrue(self.validator.check_extension("MESSAGE.EML"))

    def test_extension_must_be_at_end(self):
        self.assertFalse(self.validator.check_extension("message.eml.exe"))
        self.assertFalse(self.validator.check_extension("eml"))

    # --- check_format ---
    def test_renamed_pdf_rejected(self):
        self.assertFalse(self.validator.check_format(b"%PDF-1.7\n\x00\x01\x02 binary data"))

    def test_from_header_alone_is_not_enough(self):
        self.assertFalse(self.validator.check_format(b"From: someone@example.com\r\n\r\nhi"))

    def test_missing_from_rejected(self):
        self.assertFalse(self.validator.check_format(b"To: a@b.com\r\nSubject: hi\r\n\r\nbody"))


if __name__ == "__main__":
    unittest.main()
