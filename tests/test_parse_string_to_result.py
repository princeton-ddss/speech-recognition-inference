from pipeline.main import parse_string_to_result


def test_parse_string():
    input_string = (
        "<|0.00|> an ofd the 16 years that we've been married.<|2.00|><|2.00|> Have you"
        " one time toldf that you liked him?<|5.00|><|5.00|> Not in those exact"
        " words.<|7.00|><|7.00|> No.<|8.00|><|8.00|> No.<|9.00|><|9.00|> Not in any"
        " words, Dad.<|10.00|><|10.00|> He said that makes me feel.<|12.00|><|13.00|>"
        " You've never told your son that you love him.<|16.00|>"
    )
    result = parse_string_to_result(input_string, "en")
    assert result


def test_parse_string_with_missing_timestamps():
    input_string = (
        " ! Good morning. I now call this meeting of the Abilene City Council order is"
        " 830 a.m. I'm going to introduce here in a moment but I'm going to ask"
        " Councilman"
    )
    result = parse_string_to_result(input_string, "en")
    assert result
