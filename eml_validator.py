"""
EmailValidator
--------------
Backend file validation for Gone Phishing.

Confirms that:
  1. a file was actually provided (and isn't empty or too large),
  2. its name ends in .eml,
  3. its contents can be parsed as an email.

If any check fails, validate_file() raises EmailValidationError with a
message that is safe to show directly to the user.
"""
from email import policy
from email.parser import BytesParser


class EmailValidationError(Exception):
    """Raised when an uploaded file fails validation. str(error) is the user-facing message."""


class EmailValidator:
    NO_FILE_MESSAGE = "Please attach a valid .eml file."
    WRONG_TYPE_MESSAGE = "Unsupported file type. Please attach a valid .eml file."
    EMPTY_FILE_MESSAGE = "The attached file is empty. Please attach a valid .eml file."
    TOO_LARGE_MESSAGE = "The attached file is too large. The maximum size is 10 MB."
    BAD_FORMAT_MESSAGE = "This file could not be read as an email. Please attach a valid .eml file."

    # An email must have a From header plus at least one of these to count as a real email
    REQUIRED_HEADER = "From"
    SUPPORTING_HEADERS = ("To", "Subject", "Date")

    def __init__(self):
        self.allowed_extension = ".eml"
        self.max_file_size = 10 * 1024 * 1024  # 10 MB, in bytes

    def validate_file(self, filename, content):
        """
        Run every check in order. Returns True if the file is valid,
        otherwise raises EmailValidationError explaining the first problem found.

        filename: the uploaded file's name (str), or None if nothing was attached
        content:  the file's raw bytes
        """
        # 1. Was a file provided at all?
        if not filename:
            raise EmailValidationError(self.NO_FILE_MESSAGE)

        # 2. Does it have the right extension?
        if not self.check_extension(filename):
            raise EmailValidationError(self.WRONG_TYPE_MESSAGE)

        # 3. Is there anything in it, and is it a reasonable size?
        if not content:
            raise EmailValidationError(self.EMPTY_FILE_MESSAGE)
        if len(content) > self.max_file_size:
            raise EmailValidationError(self.TOO_LARGE_MESSAGE)

        # 4. Can the contents be parsed as an email?
        if not self.check_format(content):
            raise EmailValidationError(self.BAD_FORMAT_MESSAGE)

        return True

    def check_extension(self, filename):
        """Return True if the filename ends with .eml (case-insensitive)."""
        return filename.strip().lower().endswith(self.allowed_extension)

    def check_format(self, content):
        """
        Return True if the bytes look like a real email.

        Python's email parser is very forgiving: it will "parse" almost anything,
        even a PDF. So parsing alone proves nothing. Instead we parse, then check
        that the result has the headers every real email has.
        """
        # Binary files (images, PDFs, executables renamed to .eml) contain null bytes;
        # real emails are text, so reject these straight away.
        if b"\x00" in content:
            return False

        try:
            message = BytesParser(policy=policy.default).parsebytes(content)
            has_from = bool(message[self.REQUIRED_HEADER])
            has_supporting = any(message[header] for header in self.SUPPORTING_HEADERS)
        except Exception:
            # Malformed headers can raise errors when read; treat that as invalid.
            return False

        return has_from and has_supporting
