import pytest

from benchmarks.benchmark_kv_quality import (
    answer_matches,
    build_needle_case,
    build_quality_cases,
    final_answer_text,
    normalize_answer,
    build_quality_matrix,
    quality_fingerprint,
)


def test_needle_case_places_fact_at_requested_fraction_and_repeats_query_terms():
    case = build_needle_case("middle", 0.5, filler_lines=20)
    lines = case.prompt.splitlines()
    needle_index = next(
        index for index, line in enumerate(lines) if case.expected in line
    )

    assert 9 <= needle_index <= 11
    assert "project ORION-middle" in lines[needle_index]
    assert "project ORION-middle" in case.prompt.splitlines()[-1]


def test_answer_matching_ignores_case_and_punctuation():
    assert normalize_answer("ZXQ-MIDDLE-7419") == "ZXQMIDDLE7419"
    assert answer_matches("`zxq-middle-7419`", "ZXQ-MIDDLE-7419")
    assert answer_matches(
        "<think>reasoning</think>\nZXQ-MIDDLE-7419", "ZXQ-MIDDLE-7419"
    )
    assert not answer_matches(
        "<think>ZXQ-MIDDLE-7419", "ZXQ-MIDDLE-7419"
    )
    assert not answer_matches(
        "The answer is ZXQ-MIDDLE-7419", "ZXQ-MIDDLE-7419"
    )
    assert not answer_matches("ZXQ-EARLY-7419", "ZXQ-MIDDLE-7419")


def test_final_answer_excludes_reasoning_text():
    assert final_answer_text("<think>ZXQ-WRONG</think>ZXQ-RIGHT") == "ZXQ-RIGHT"
    assert final_answer_text("<think>unfinished") == ""


def test_needle_case_validates_dimensions():
    with pytest.raises(ValueError, match="fraction"):
        build_needle_case("bad", 1.1)
    with pytest.raises(ValueError, match="eight"):
        build_needle_case("bad", 0.5, filler_lines=4)


def test_quality_case_matrix_repeats_each_needle_position():
    cases = build_quality_cases(20, repeats_per_position=3)

    assert len(cases) == 9
    assert [case.needle_fraction for case in cases].count(0.5) == 3
    assert len({case.expected for case in cases}) == 9


def test_needle_prompt_disables_thinking_for_strict_retrieval_evaluation():
    case = build_needle_case("early", 0.2, filler_lines=20)

    assert "/no_think" in case.prompt.splitlines()[-2]


def test_quality_matrix_has_diverse_reproducible_cases_and_random_answers():
    cases = build_quality_matrix(seed=19)
    assert len(cases) == 216
    assert {case.task for case in cases} == {"literal", "paraphrase", "distractors", "two_hop"}
    assert cases == build_quality_matrix(seed=19)
    assert quality_fingerprint(cases) != quality_fingerprint(build_quality_matrix(seed=20))
    for case in cases:
        assert case.expected not in case.case_id
        assert case.prompt.count(case.expected) == 1
        assert case.expected not in case.prompt.split("/no_think")[-1]
