
from utils.datagenerator import WikiReadingDataGenerator
import pytest


def test_demodatagenerator():

    generator = WikiReadingDataGenerator()
    # test the demo_vectorizefunction for the generator
    x = generator.generate(['data/test-00000-of-00015.json'], 20)
    c = 0
    for i in x:
        c += 1
        assert len(i) == 2
        assert len(i[1]) == 20

        if c == 10:
            break

    x = generator.generate(['data/test-00000-of-00015.json'], 5)

    for i in x:
        c += 1
        assert len(i) == 2
        assert len(i[1]) == 5

        if c == 10:
            break










