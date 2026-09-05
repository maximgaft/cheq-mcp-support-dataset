"""The host-eval grader, checked on deliberately right and deliberately wrong answers.

A grader that only looks for expected tokens passes swapped tables and refusals
followed by invented trends. These fixtures pin the rules that stop that, and the
warning list that surfaces numbers no tool returned.
"""

import importlib

h = importlib.import_module("evals.host_eval")

TAGS = {"kind": "sql", "expect_rows": [["IT", 7446], ["Tech Support", 7413], ["Performance", 6642]]}
REFUSE_TIME = {"kind": "refuse", "expect": "no timestamps"}
SCHEMA_TEXT = ["{'tables': {'tickets': {'rows': 40047}}, 'notes': ['agrees 75% of the time']}"]


def test_table_values_must_sit_next_to_their_own_label():
    right = "Top tags: IT (7,446), Tech Support (7,413), Performance (6,642)."
    swapped = "IT: 6,642; Tech Support: 7,446; Performance: 7,413."
    assert h.grade(TAGS, right, ["run_sql"], SCHEMA_TEXT)[0] is True
    passed, reason, _ = h.grade(TAGS, swapped, ["run_sql"], SCHEMA_TEXT)
    assert passed is False and "not next to their label" in reason


def test_short_labels_match_whole_words_only():
    # "it" in prose is not the IT tag
    assert h.row_present("Tech Support: 7,446; it also lists Performance", ["IT", 7446], ["Tech Support", "Performance"]) is False
    assert h.row_present("| IT | 7,446 |", ["IT", 7446], ["Tech Support", "Performance"]) is True


def test_label_that_prefixes_another_label_is_not_confused_with_it():
    case = {"kind": "sql", "expect_rows": [["IT", 7446], ["IT Support", 4768]]}
    assert h.grade(case, "| IT | 7,446 |\n| IT Support | 4,768 |", ["run_sql"], [])[0] is True
    assert h.grade(case, "| IT | 4,768 |\n| IT Support | 7,446 |", ["run_sql"], [])[0] is False


def test_unlabelled_rows_only_need_their_values():
    case = {"kind": "sql", "expect": [4858, 12230, 39.7]}
    assert h.grade(case, "39.72% of German tickets (4,858 of 12,230) are high priority.", ["run_sql"], [])[0] is True


def test_float_key_accepts_a_more_precise_answer_that_rounds_to_it():
    case = {"kind": "sql", "expect": [39.7]}
    assert h.grade(case, "39.72%", ["run_sql"], [])[0] is True
    assert h.grade(case, "39.65%", ["run_sql"], [])[0] is False


def test_refusal_must_lead_and_must_not_assert_a_trend():
    good = "This can't be answered: there are no timestamps. The table has 12 columns and 40,047 rows."
    trend = "No timestamps are available. Monthly volume increased by 50 percent."
    higher = "There are no dates in the data. Still, volume is 12% higher than last quarter."
    answered_first = "Volume rose 8% month over month, although the data has no timestamps."
    assert h.grade(REFUSE_TIME, good, ["get_schema"], SCHEMA_TEXT)[0] is True
    assert h.grade(REFUSE_TIME, trend, ["get_schema"], SCHEMA_TEXT)[1] == "refused, then asserted a trend"
    assert h.grade(REFUSE_TIME, higher, ["get_schema"], SCHEMA_TEXT)[1] == "refused, then asserted a trend"
    assert h.grade(REFUSE_TIME, answered_first, ["get_schema"], SCHEMA_TEXT)[1] == "did not lead with a refusal"
    # an answer first, hedge later, with no trend verb at all
    sneaky = "Technical Support leads with 11,718 tickets, though there is no date column to trend it."
    assert h.grade(REFUSE_TIME, sneaky, ["get_schema"], ["11718"])[1] == "did not lead with a refusal"
    # a refusal that restates the question with a trend verb is still a refusal
    restated = "This can't be answered: we cannot tell whether volume increased or decreased over the last 6 months because there are no timestamps."
    assert h.grade(REFUSE_TIME, restated, ["get_schema"], SCHEMA_TEXT)[0] is True
    assert h.grade(REFUSE_TIME, "**No.** There are no timestamps in the data.", ["get_schema"], SCHEMA_TEXT)[0] is True
    assert h.grade(REFUSE_TIME, "Timestamps are absent from this dataset, so no.", ["get_schema"], SCHEMA_TEXT)[0] is True


def test_unsourced_numbers_are_listed_but_derived_ones_are_not():
    tool = ["[[355]] [[4009]] " + " ".join(str(n) for n in range(100, 1300, 7))]   # a schema dump's worth of numbers
    answer = "355 tickets. Billing holds 4,009 in total; 1,478 of them are high priority, so the share is 24%."
    passed, _, warnings = h.grade({"kind": "sql", "expect": [355]}, answer, ["run_sql"], tool)
    assert passed is True and warnings == ["1,478", "24%"]     # 24% is 355/1,478, and 1,478 is unsourced
    correct = "355 tickets. Billing holds 4,009 in total; 1,185 of them are high priority, so the share is 29.96%."
    assert h.unsourced_numbers(correct, ["355 4009 1185"], []) == []   # 29.96% = 355/1,185, both cited and sourced
    # a share over a three-part total the model summed itself
    shares = "dead_end 19,781 (49.80%), actionable_ask 15,031 (37.84%), resolved 4,910 (12.36%)."
    assert h.unsourced_numbers(shares, ["19781 15031 4910"], []) == []
    derived = "19,781 of 39,722 labelled replies (49.80%) are dead ends; 40,047 minus 325 unlabelled."
    assert h.unsourced_numbers(derived, ["19781 40047 325 39722"], []) == []
    assert h.unsourced_numbers("The table has 12 columns.", [], []) == []          # small item counts are not claims
    assert h.unsourced_numbers("39.72% are high priority", ["39.72"], []) == []     # tool-returned, more precise
    assert h.unsourced_numbers("about 39.7%", ["39.72"], []) == []                   # a rounding of a tool number is sourced


def test_paraphrased_label_is_reported_not_failed():
    case = {"kind": "sql", "expect": ["en", 27817]}
    passed, reason, _ = h.grade(case, "English has more tickets: 27,817 (69.5%) vs 12,230 German.", ["run_sql"], ["27817 12230"])
    assert passed is True and "paraphrased" in reason


def test_no_tool_call_fails_even_with_right_numbers():
    assert h.grade({"kind": "sql", "expect": [40047]}, "40,047 tickets.", [], [])[1] == "right numbers but no tool was called"
