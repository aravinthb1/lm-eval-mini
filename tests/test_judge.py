import pytest

from evalkit.judge import JudgeParseError, parse_judge_response


def test_plain_json():
    r = parse_judge_response('{"score": 5, "reason": "good"}')
    assert (r.score, r.reason) == (5, "good")


def test_markdown_fenced_json():
    assert parse_judge_response('```json\n{"score": 3, "reason": "ok"}\n```').score == 3


def test_json_surrounded_by_chatter():
    assert parse_judge_response('Sure! Here you go: {"score": 2, "reason": "meh"} Hope that helps.').score == 2


@pytest.mark.parametrize("bad", [
    "no json here",
    '{"score": 9, "reason": "x"}',
    '{"score": 0, "reason": "x"}',
    '{"score": "high", "reason": "x"}',
    '{"score": true, "reason": "x"}',
    '{"reason": "missing score"}',
    "{not valid json}",
])
def test_bad_replies_raise(bad):
    with pytest.raises(JudgeParseError):
        parse_judge_response(bad)
