import json
import os
import tempfile
import time
import unittest

from scripts.chat_history import ChatEntry, ChatHistoryStore, record_session_result


class TestChatHistoryStore(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        os.unlink(self._tmp.name)  # let the store create it fresh
        self.store = ChatHistoryStore(storage_path=self._tmp.name)

    def tearDown(self):
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)

    # ------------------------------------------------------------------
    # record / get_recent
    # ------------------------------------------------------------------

    def test_record_and_get_recent(self):
        entry = ChatEntry(timestamp=time.time(), user_input="hello", result="world", status="SUCCESS")
        self.store.record(entry)
        recent = self.store.get_recent(limit=5)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].user_input, "hello")
        self.assertEqual(recent[0].result, "world")
        self.assertEqual(recent[0].status, "SUCCESS")

    def test_get_recent_newest_first(self):
        t = time.time()
        self.store.record(ChatEntry(timestamp=t, user_input="first", result="a", status="SUCCESS"))
        self.store.record(ChatEntry(timestamp=t + 1, user_input="second", result="b", status="SUCCESS"))
        recent = self.store.get_recent(limit=5)
        self.assertEqual(recent[0].user_input, "second")
        self.assertEqual(recent[1].user_input, "first")

    def test_get_recent_respects_limit(self):
        for i in range(5):
            self.store.record(ChatEntry(timestamp=float(i), user_input=f"q{i}", result="r", status="SUCCESS"))
        recent = self.store.get_recent(limit=3)
        self.assertEqual(len(recent), 3)

    def test_empty_store_returns_empty_list(self):
        self.assertEqual(self.store.get_recent(), [])

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def test_search_matches_user_input(self):
        self.store.record(ChatEntry(timestamp=1.0, user_input="What is the temperature?", result="300K", status="SUCCESS"))
        self.store.record(ChatEntry(timestamp=2.0, user_input="List all valves", result="V1, V2", status="SUCCESS"))
        results = self.store.search("temperature")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].user_input, "What is the temperature?")

    def test_search_matches_result_field(self):
        self.store.record(ChatEntry(timestamp=1.0, user_input="status check", result="FAULT_CODE_42", status="SUCCESS"))
        results = self.store.search("FAULT_CODE_42")
        self.assertEqual(len(results), 1)

    def test_search_case_insensitive(self):
        self.store.record(ChatEntry(timestamp=1.0, user_input="Check Pressure", result="ok", status="SUCCESS"))
        self.assertEqual(len(self.store.search("check pressure")), 1)
        self.assertEqual(len(self.store.search("CHECK PRESSURE")), 1)

    def test_search_empty_query_returns_recent(self):
        for i in range(3):
            self.store.record(ChatEntry(timestamp=float(i), user_input=f"q{i}", result="r", status="SUCCESS"))
        self.assertEqual(len(self.store.search("")), 3)

    def test_search_no_match_returns_empty(self):
        self.store.record(ChatEntry(timestamp=1.0, user_input="hello", result="world", status="SUCCESS"))
        self.assertEqual(self.store.search("zzz_no_match"), [])

    def test_search_respects_limit(self):
        for i in range(5):
            self.store.record(ChatEntry(timestamp=float(i), user_input=f"common term {i}", result="r", status="SUCCESS"))
        results = self.store.search("common term", limit=2)
        self.assertEqual(len(results), 2)

    def test_search_returns_newest_first(self):
        self.store.record(ChatEntry(timestamp=1.0, user_input="old query", result="a", status="SUCCESS"))
        self.store.record(ChatEntry(timestamp=2.0, user_input="new query", result="b", status="SUCCESS"))
        results = self.store.search("query")
        self.assertEqual(results[0].user_input, "new query")

    # ------------------------------------------------------------------
    # record_session_result helper
    # ------------------------------------------------------------------

    def test_record_session_result_success(self):
        record_session_result(self.store, "sky color?", "The sky is blue.", session_id="s1")
        entries = self.store.get_recent()
        self.assertEqual(entries[0].status, "SUCCESS")
        self.assertEqual(entries[0].session_id, "s1")

    def test_record_session_result_escalated(self):
        result = {"status": "ESCALATED_TO_HUMAN", "last_draft": "draft", "conflict_summary": "x", "action_required": "y"}
        record_session_result(self.store, "q", result)
        self.assertEqual(self.store.get_recent()[0].status, "ESCALATED_TO_HUMAN")

    def test_record_session_result_insufficient(self):
        record_session_result(self.store, "q", "INSUFFICIENT VARIABLES: Missing Time.")
        self.assertEqual(self.store.get_recent()[0].status, "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
