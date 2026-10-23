"""Phase 10 tests: CHROMA_URL switches RAGSystem between the embedded
persistent client and the Chroma HTTP server used by docker compose.

No live Chroma, network, or AI: chromadb clients are patched out.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pathlib import Path

import phase1
from phase1 import PROJECT_ROOT, RAGSystem, build_chroma_client


class BuildChromaClientTests(unittest.TestCase):
    def test_embedded_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHROMA_URL", None)
            with patch("phase1.chromadb.PersistentClient") as persistent, \
                    patch("phase1.chromadb.HttpClient") as http:
                build_chroma_client("chroma_data")
        persistent.assert_called_once()
        http.assert_not_called()
        self.assertTrue(
            persistent.call_args.kwargs["path"].endswith("chroma_data")
        )

    def test_embedded_anchors_relative_path_to_project(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHROMA_URL", None)
            with patch("phase1.chromadb.PersistentClient") as persistent:
                build_chroma_client("some/relative")
        expected = str((PROJECT_ROOT / "some" / "relative").resolve())
        self.assertEqual(persistent.call_args.kwargs["path"], expected)

    def test_embedded_keeps_absolute_path(self):
        absolute = str(Path(PROJECT_ROOT).resolve())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHROMA_URL", None)
            with patch("phase1.chromadb.PersistentClient") as persistent:
                build_chroma_client(absolute)
        self.assertEqual(persistent.call_args.kwargs["path"], absolute)

    def test_blank_url_stays_embedded(self):
        with patch.dict(os.environ, {"CHROMA_URL": "   "}):
            with patch("phase1.chromadb.PersistentClient") as persistent, \
                    patch("phase1.chromadb.HttpClient") as http:
                build_chroma_client("chroma_data")
        persistent.assert_called_once()
        http.assert_not_called()

    def test_server_mode_uses_http_client(self):
        with patch.dict(os.environ, {"CHROMA_URL": "http://chroma:8000"}):
            with patch("phase1.chromadb.HttpClient") as http, \
                    patch("phase1.chromadb.PersistentClient") as persistent:
                build_chroma_client("chroma_data")
        http.assert_called_once_with(host="chroma", port=8000, ssl=False)
        persistent.assert_not_called()

    def test_server_mode_defaults_to_port_8000(self):
        with patch.dict(os.environ, {"CHROMA_URL": "http://chroma"}):
            with patch("phase1.chromadb.HttpClient") as http:
                build_chroma_client("chroma_data")
        http.assert_called_once_with(host="chroma", port=8000, ssl=False)

    def test_server_mode_https_sets_ssl(self):
        with patch.dict(
            os.environ, {"CHROMA_URL": "https://chroma.example:8443"}
        ):
            with patch("phase1.chromadb.HttpClient") as http:
                build_chroma_client("chroma_data")
        http.assert_called_once_with(
            host="chroma.example", port=8443, ssl=True
        )


class EmbedDocumentsTests(unittest.TestCase):
    def test_each_chunk_is_sent_as_its_own_content(self):
        fake_client = Mock()
        fake_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.1, 0.2]),
                SimpleNamespace(values=[0.3, 0.4]),
            ]
        )
        with patch("phase1.get_gemini_client", return_value=fake_client):
            embeddings = phase1.embed_documents(["first", "second"])

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])
        contents = fake_client.models.embed_content.call_args.kwargs["contents"]
        self.assertEqual(
            [content.parts[0].text for content in contents],
            ["first", "second"],
        )

    def test_mismatched_embedding_count_is_rejected(self):
        fake_client = Mock()
        fake_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2])]
        )
        with patch("phase1.get_gemini_client", return_value=fake_client):
            with self.assertRaisesRegex(ValueError, "1 embeddings for 2"):
                phase1.embed_documents(["first", "second"])


class RAGSystemClientTests(unittest.TestCase):
    def test_rag_system_uses_build_chroma_client(self):
        fake_client = Mock()
        with patch(
            "phase1.build_chroma_client", return_value=fake_client
        ) as builder:
            rag = RAGSystem(
                collection_name="learning_rag",
                chroma_path="chroma_data",
            )
        builder.assert_called_once_with("chroma_data")
        fake_client.get_or_create_collection.assert_called_once()
        self.assertEqual(
            fake_client.get_or_create_collection.call_args.kwargs["name"],
            "learning_rag",
        )
        self.assertIs(rag.client, fake_client)


if __name__ == "__main__":
    unittest.main()
