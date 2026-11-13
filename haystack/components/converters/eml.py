# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)

_HEADERS = ("From", "To", "Cc", "Bcc", "Subject")
_ENCRYPTED_CONTENT_TYPES = ("application/pkcs7-mime", "application/x-pkcs7-mime", "multipart/encrypted")


@component
class EMLToDocument:
    """
    Converts email files in the standard .eml format (RFC 822 / MIME) into Haystack Documents.

    .eml is the format used when saving emails from Gmail, Thunderbird, Apple Mail, and most other email clients.
    This component extracts the email headers (sender, recipients, CC, BCC, subject) and the body into a Document,
    using the same layout as `MSGToDocument`. The plain text body is preferred. If an email has only an HTML body,
    the HTML is kept as it is. Any file attachments are returned as ByteStream objects. An attached email is returned
    as a .eml ByteStream, so it can be converted again with this component.

    It uses only the Python standard library, so no extra packages are needed.

    ### Example Usage

    ```python
    from haystack.components.converters.eml import EMLToDocument
    from datetime import datetime

    converter = EMLToDocument()
    results = converter.run(sources=["test/test_files/eml/sample.eml"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    attachments = results["attachments"]
    print(documents[0].content)
    # From: Alice Example <alice@example.com>
    # To: Bob Example <bob@example.com>
    # Cc: carol@example.com
    # Subject: Q3 report – final
    #
    # Hi Bob,
    # ...
    ```
    """

    def __init__(self, store_full_path: bool = False) -> None:
        """
        Creates an EMLToDocument component.

        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        self.store_full_path = store_full_path

    @staticmethod
    def _get_text(part: EmailMessage) -> str:
        """
        Decodes the text of a body part.

        Falls back to UTF-8 with replacement characters when the email declares an unknown charset.
        CRLF line endings, which .eml files usually use, are converted to LF.

        :param part: A text/plain or text/html part of the email.
        :returns: The decoded text.
        """
        try:
            text = part.get_content()
        except (LookupError, UnicodeDecodeError):
            payload = part.get_payload(decode=True)
            text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else ""
        return text.replace("\r\n", "\n")

    @staticmethod
    def _get_attachments(msg: EmailMessage) -> list[ByteStream]:
        """
        Extracts the file attachments of the email.

        :param msg: The parsed email.
        :returns: A list of ByteStream objects, one for each attachment.
        """
        attachments = []
        for index, part in enumerate(msg.iter_attachments()):
            content_type = part.get_content_type()
            if content_type == "message/rfc822":
                # An attached email: keep it as a .eml file so it can be converted again.
                inner_message = part.get_content()
                data = inner_message.as_bytes() if isinstance(inner_message, EmailMessage) else None
                default_name = f"attachment_{index}.eml"
            else:
                # Nested multipart containers have no payload to decode and are skipped.
                payload = part.get_payload(decode=True)
                data = payload if isinstance(payload, bytes) else None
                default_name = f"attachment_{index}"

            if data is None:
                continue

            attachments.append(
                ByteStream(data=data, meta={"file_path": part.get_filename() or default_name}, mime_type=content_type)
            )
        return attachments

    def _convert(self, data: bytes) -> tuple[str, list[ByteStream]]:
        """
        Converts the EML file content into text and extracts any attachments.

        :param data: The EML file content as bytes.
        :returns: A tuple containing the extracted email text and a list of ByteStream objects for attachments.
        :raises ValueError: If the content is not an email or if the email is encrypted.
        """
        msg = BytesParser(policy=policy.default).parsebytes(data)
        if not isinstance(msg, EmailMessage) or not any(msg[header] for header in ("From", "Date", "Subject")):
            raise ValueError("The file is not a valid EML file: no 'From', 'Date', or 'Subject' header found.")
        if msg.get_content_type() in _ENCRYPTED_CONTENT_TYPES:
            raise ValueError("The EML file is encrypted and cannot be read.")

        txt = ""

        # Sender, recipients, and subject
        for header in _HEADERS:
            value = msg[header]
            if value is not None and str(value).strip():
                txt += f"{header}: {value}\n"

        # Body: prefer plain text, fall back to HTML
        body = msg.get_body(preferencelist=("plain", "html"))
        if isinstance(body, EmailMessage):
            txt += "\n" + self._get_text(body)

        return txt, self._get_attachments(msg)

    @component.output_types(documents=list[Document], attachments=list[ByteStream])
    def run(
        self, sources: list[str | Path | ByteStream], meta: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> dict[str, list[Document] | list[ByteStream]]:
        """
        Converts EML files to Documents.

        :param sources:
            List of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will
            be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: Created Documents.
            - `attachments`: Created ByteStream objects from file attachments.
        """
        if len(sources) == 0:
            return {"documents": [], "attachments": []}

        documents = []
        all_attachments = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                text, attachments = self._convert(bytestream.data)
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path and "file_path" in bytestream.meta:
                merged_metadata["file_path"] = os.path.basename(bytestream.meta["file_path"])

            documents.append(Document(content=text, meta=merged_metadata))
            for attachment in attachments:
                attachment_meta = {**merged_metadata, "file_path": attachment.meta["file_path"]}
                if "file_path" in merged_metadata:
                    attachment_meta["parent_file_path"] = merged_metadata["file_path"]
                all_attachments.append(
                    ByteStream(data=attachment.data, meta=attachment_meta, mime_type=attachment.mime_type)
                )

        return {"documents": documents, "attachments": all_attachments}
