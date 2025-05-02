from speech_recognition_inference.pipeline import Transcription, TranscriptChunk


def test_parse_string():
    raw_string = (
        "<|0.00|> an of the 16 years that we've been married.<|2.00|><|2.00|> Have you"
        " one time told me that you liked him?<|5.00|><|5.00|> Not in those exact"
        " words.<|7.00|><|7.00|> No.<|8.00|><|8.00|> No.<|9.00|><|9.00|> Not in any"
        " words, Dad.<|10.00|><|10.00|> He said that makes me feel.<|12.00|><|13.00|>"
        " You've never told your son that you love him.<|16.00|>"
    )
    transcript = Transcription.from_string(raw_string, language="en")
    assert transcript == Transcription(
        language="en",
        text=(
            " an of the 16 years that we've been married. Have you one time told me"
            " that you liked him? Not in those exact words. No. No. Not in any words,"
            " Dad. He said that makes me feel. You've never told your son that you love"
            " him."
        ),
        chunks=[
            TranscriptChunk(
                text=" an of the 16 years that we've been married.",
                timestamp=(0.0, 2.0),
            ),
            TranscriptChunk(
                text=" Have you one time told me that you liked him?",
                timestamp=(2.0, 5.0),
            ),
            TranscriptChunk(text=" Not in those exact words.", timestamp=(5.0, 7.0)),
            TranscriptChunk(text=" No.", timestamp=(7.0, 8.0)),
            TranscriptChunk(text=" No.", timestamp=(8.0, 9.0)),
            TranscriptChunk(text=" Not in any words, Dad.", timestamp=(9.0, 10.0)),
            TranscriptChunk(
                text=" He said that makes me feel.", timestamp=(10.0, 12.0)
            ),
            TranscriptChunk(
                text=" You've never told your son that you love him.",
                timestamp=(13.0, 16.0),
            ),
        ],
    )


def test_parse_string_with_missing_timestamps():
    raw_string = (
        " ! Good morning. I now call this meeting of the Abilene City Council order is"
        " 830 a.m. I'm going to introduce here in a moment but I'm going to ask"
        " Councilman"
    )
    transcript = Transcription.from_string(raw_string, "en")
    assert transcript == Transcription(
        text=(
            "Good morning. I now call this meeting of the Abilene City Council order is"
            " 830 a.m. I'm going to introduce here in a moment but I'm going to ask"
            " Councilman"
        ),
        language="en",
        chunks=[
            TranscriptChunk(
                text=(
                    " ! Good morning. I now call this meeting of the Abilene City"
                    " Council order is 830 a.m. I'm going to introduce here in a moment"
                    " but I'm going to ask Councilman"
                ),
                timestamp=(0.0, 30.0),
            )
        ],
    )
