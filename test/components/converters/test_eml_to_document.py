# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from email.message import EmailMessage

import pytest

from haystack import Document, Pipeline
from haystack.components.converters.eml import EMLToDocument
from haystack.components.routers import FileTypeRouter
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream

SAMPLE_ATTACHMENT_DATA = b"quarter,revenue\nQ1,95\nQ2,100\nQ3,112\n"
SAMPLE_CONTENT = (
    "From: Alice Example <alice@example.com>\n"
    "To: Bob Example <bob@example.com>\n"
    "Cc: carol@example.com\n"
    "Subject: Q3 report – final\n"
    "\n"
    "Hi Bob,\n"
    "\n"
    "The Q3 report is attached. Revenue grew 12% to 112 million.\n"
    "\n"
    "Alice\n"
)


def _make_email(**headers: str) -> EmailMessage:
    msg = EmailMessage()
    for name, value in headers.items():
        msg[name] = value
    return msg


@pytest.fixture
def simple_email() -> EmailMessage:
    msg = _make_email(From="alice@example.com", To="bob@example.com", Subject="Hello")
    msg.set_content("Just a short note.")
    return msg


class TestEMLToDocument:
    def test_init_default(self):
        converter = EMLToDocument()
        assert converter.store_full_path is False

    def test_init_store_full_path(self):
        converter = EMLToDocument(store_full_path=True)
        assert converter.store_full_path is True

    def test_serialization_round_trip(self):
        converter = EMLToDocument(store_full_path=True)
        data = component_to_dict(converter, "converter")
        assert data == {
            "type": "haystack.components.converters.eml.EMLToDocument",
            "init_parameters": {"store_full_path": True},
        }
        new_converter = component_from_dict(EMLToDocument, data, "converter")
        assert new_converter.store_full_path is True

    def test_run(self, test_files_path):
        converter = EMLToDocument(store_full_path=True)
        path = test_files_path / "eml" / "sample.eml"
        result = converter.run(sources=[path], meta={"date_added": "2021-09-01T00:00:00"})

        assert len(result["documents"]) == 1
        document = result["documents"][0]
        assert isinstance(document, Document)
        assert document.content == SAMPLE_CONTENT
        assert document.meta == {"date_added": "2021-09-01T00:00:00", "file_path": str(path)}

        assert len(result["attachments"]) == 1
        attachment = result["attachments"][0]
        assert isinstance(attachment, ByteStream)
        assert attachment.mime_type == "text/csv"
        assert attachment.data == SAMPLE_ATTACHMENT_DATA
        assert attachment.meta == {
            "date_added": "2021-09-01T00:00:00",
            "file_path": "q3.csv",
            "parent_file_path": str(path),
        }

    def test_run_stores_only_file_name_by_default(self, test_files_path):
        result = EMLToDocument().run(sources=[test_files_path / "eml" / "sample.eml"])
        assert result["documents"][0].meta == {"file_path": "sample.eml"}
        assert result["attachments"][0].meta == {"file_path": "q3.csv", "parent_file_path": "sample.eml"}

    def test_run_bytestream_without_file_path(self, test_files_path):
        """A bare ByteStream source (no 'file_path' in its meta) must not raise a KeyError."""
        data = (test_files_path / "eml" / "sample.eml").read_bytes()
        result = EMLToDocument().run(sources=[ByteStream(data=data)], meta={"date_added": "2021-09-01T00:00:00"})

        assert result["documents"][0].meta == {"date_added": "2021-09-01T00:00:00"}
        assert result["attachments"][0].meta == {"date_added": "2021-09-01T00:00:00", "file_path": "q3.csv"}
        assert "parent_file_path" not in result["attachments"][0].meta

    def test_run_merges_bytestream_meta_and_meta_list(self, simple_email):
        sources = [
            ByteStream(data=bytes(simple_email), meta={"file_path": "a.eml", "source": "inbox"}),
            ByteStream(data=bytes(simple_email), meta={"file_path": "b.eml"}),
        ]
        result = EMLToDocument().run(sources=sources, meta=[{"source": "archive"}, {"lang": "en"}])

        assert [doc.meta for doc in result["documents"]] == [
            {"file_path": "a.eml", "source": "archive"},
            {"file_path": "b.eml", "lang": "en"},
        ]

    def test_run_simple_email_without_attachments(self, simple_email):
        result = EMLToDocument().run(sources=[ByteStream(data=bytes(simple_email))])

        assert result["documents"][0].content == (
            "From: alice@example.com\nTo: bob@example.com\nSubject: Hello\n\nJust a short note.\n"
        )
        assert result["attachments"] == []

    def test_run_skips_missing_and_blank_headers(self):
        msg = _make_email(From="alice@example.com", Cc="   ", Subject="No recipients")
        msg.set_content("Body")
        content = EMLToDocument().run(sources=[ByteStream(data=bytes(msg))])["documents"][0].content

        assert content == "From: alice@example.com\nSubject: No recipients\n\nBody\n"

    def test_run_includes_bcc(self):
        msg = _make_email(From="alice@example.com", To="bob@example.com", Bcc="dave@example.com", Subject="Hi")
        msg.set_content("Body")
        content = EMLToDocument().run(sources=[ByteStream(data=bytes(msg))])["documents"][0].content

        assert "Bcc: dave@example.com\n" in content

    def test_run_prefers_plain_text_over_html(self):
        msg = _make_email(From="alice@example.com", Subject="Both bodies")
        msg.set_content("Plain body")
        msg.add_alternative("<p>HTML body</p>", subtype="html")
        content = EMLToDocument().run(sources=[ByteStream(data=bytes(msg))])["documents"][0].content

        assert content.endswith("\nPlain body\n")
        assert "HTML body" not in content

    def test_run_html_only_email_keeps_html(self):
        msg = _make_email(From="it@example.com", Subject="Policy")
        msg.set_content("<p>Passwords must be reset every 90 days.</p>", subtype="html")
        content = EMLToDocument().run(sources=[ByteStream(data=bytes(msg))])["documents"][0].content

        assert content == "From: it@example.com\nSubject: Policy\n\n<p>Passwords must be reset every 90 days.</p>\n"

    def test_run_email_without_body(self):
        msg = _make_email(From="alice@example.com", Subject="Only an attachment")
        msg.add_attachment(b"data", maintype="application", subtype="octet-stream", filename="data.bin")
        result = EMLToDocument().run(sources=[ByteStream(data=bytes(msg))])

        assert result["documents"][0].content == "From: alice@example.com\nSubject: Only an attachment\n"
        assert result["attachments"][0].meta == {"file_path": "data.bin"}

    def test_run_normalizes_crlf_line_endings(self, test_files_path):
        data = (test_files_path / "eml" / "sample.eml").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        result = EMLToDocument().run(sources=[ByteStream(data=data)])

        assert result["documents"][0].content == SAMPLE_CONTENT
        assert result["attachments"][0].data == SAMPLE_ATTACHMENT_DATA

    def test_run_unknown_charset_falls_back_to_utf8(self):
        data = (
            b"From: alice@example.com\r\n"
            b"Subject: Strange charset\r\n"
            b'Content-Type: text/plain; charset="not-a-real-charset"\r\n'
            b"\r\n"
            b"Caf\xc3\xa9 at noon\r\n"
        )
        content = EMLToDocument().run(sources=[ByteStream(data=data)])["documents"][0].content

        assert content == "From: alice@example.com\nSubject: Strange charset\n\nCafé at noon\n"

    def test_run_attachment_without_file_name(self):
        msg = _make_email(From="alice@example.com", Subject="Unnamed attachment")
        msg.set_content("See attached.")
        data = b"raw bytes"
        msg.add_attachment(data, maintype="application", subtype="octet-stream")
        attachment = EMLToDocument().run(sources=[ByteStream(data=bytes(msg))])["attachments"][0]

        assert attachment.meta == {"file_path": "attachment_0"}
        assert attachment.mime_type == "application/octet-stream"
        assert attachment.data == data

    def test_run_skips_nested_multipart_container(self):
        data = (
            b"From: alice@example.com\r\n"
            b"Subject: Nested\r\n"
            b'Content-Type: multipart/mixed; boundary="outer"\r\n'
            b"\r\n"
            b"--outer\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n"
            b"Body\r\n"
            b"--outer\r\n"
            b'Content-Type: multipart/mixed; boundary="inner"\r\n'
            b"\r\n"
            b"--inner\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n"
            b"Nested part\r\n"
            b"--inner--\r\n"
            b"--outer--\r\n"
        )
        result = EMLToDocument().run(sources=[ByteStream(data=data)])

        assert result["documents"][0].content == "From: alice@example.com\nSubject: Nested\n\nBody"
        assert result["attachments"] == []

    def test_run_attached_email_is_returned_as_eml(self, simple_email):
        msg = _make_email(From="carol@example.com", Subject="Fwd: Hello")
        msg.set_content("Forwarding this.")
        msg.add_attachment(simple_email)
        converter = EMLToDocument()
        attachments = converter.run(sources=[ByteStream(data=bytes(msg), meta={"file_path": "fwd.eml"})])["attachments"]

        assert len(attachments) == 1
        assert attachments[0].mime_type == "message/rfc822"
        assert attachments[0].meta == {"file_path": "attachment_0.eml", "parent_file_path": "fwd.eml"}

        inner = converter.run(sources=attachments)["documents"]
        assert (
            inner[0].content == "From: alice@example.com\nTo: bob@example.com\nSubject: Hello\n\nJust a short note.\n"
        )

    def test_run_wrong_file_type(self, test_files_path, caplog):
        with caplog.at_level(logging.WARNING):
            result = EMLToDocument().run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])

        assert result["documents"] == []
        assert result["attachments"] == []
        assert "not a valid EML file" in caplog.text

    def test_run_encrypted_email(self, caplog):
        msg = _make_email(From="alice@example.com", Subject="Secret")
        msg.set_content(b"encrypted bytes", maintype="application", subtype="pkcs7-mime")
        with caplog.at_level(logging.WARNING):
            result = EMLToDocument().run(sources=[ByteStream(data=bytes(msg))])

        assert result["documents"] == []
        assert "encrypted" in caplog.text

    def test_run_missing_file_is_skipped(self, test_files_path, caplog):
        sources = [test_files_path / "eml" / "does_not_exist.eml", test_files_path / "eml" / "sample.eml"]
        with caplog.at_level(logging.WARNING):
            result = EMLToDocument().run(sources=sources)

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["file_path"] == "sample.eml"
        assert "Could not read" in caplog.text

    def test_run_empty_sources(self):
        result = EMLToDocument().run(sources=[])
        assert result == {"documents": [], "attachments": []}

    def test_run_in_pipeline_with_file_type_router(self, test_files_path):
        pipeline = Pipeline()
        pipeline.add_component("router", FileTypeRouter(mime_types=["message/rfc822"]))
        pipeline.add_component("converter", EMLToDocument())
        pipeline.connect("router.message/rfc822", "converter.sources")

        result = pipeline.run({"router": {"sources": [test_files_path / "eml" / "sample.eml"]}})

        assert result["converter"]["documents"][0].content == SAMPLE_CONTENT
        assert result["converter"]["attachments"][0].meta["file_path"] == "q3.csv"
