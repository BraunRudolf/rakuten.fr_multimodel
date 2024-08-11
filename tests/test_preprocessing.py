import pytest

from src.dataset.preprocess import remove_html, remove_punction, remove_white_space, to_lower


@pytest.fixture()
def original_text():
    string = " Desert  Child - Jeu En Téléchargement <div> <p><strong>NOTE :</strong> Un compte Steam et une connexion internet sont nécessaires pour activer télécharger et utiliser ce produit.</p> À propos du jeu  <p>Vous êtes  ruiné vous avez$   "
    return string


@pytest.fixture()
def lower_text():
    string = " desert  child - jeu en téléchargement <div> <p><strong>note :</strong> un compte steam et une connexion internet sont nécessaires pour activer télécharger et utiliser ce produit.</p> à propos du jeu  <p>vous êtes  ruiné vous avez$   "
    return string


@pytest.fixture()
def no_punctuation_text():
    string = " Desert  Child  Jeu En Téléchargement div pstrongNOTE strong Un compte Steam et une connexion internet sont nécessaires pour activer télécharger et utiliser ce produitp À propos du jeu  pVous êtes  ruiné vous avez   "
    return string


@pytest.fixture()
def no_html_text():
    string = " Desert  Child - Jeu En Téléchargement  NOTE : Un compte Steam et une connexion internet sont nécessaires pour activer télécharger et utiliser ce produit. À propos du jeu  Vous êtes  ruiné vous avez$   "
    return string


@pytest.fixture()
def no_white_space_text():
    string = "Desert Child - Jeu En Téléchargement <div> <p><strong>NOTE :</strong> Un compte Steam et une connexion internet sont nécessaires pour activer télécharger et utiliser ce produit.</p> À propos du jeu <p>Vous êtes ruiné vous avez$"
    return string


@pytest.fixture()
def leading_white_space_text():
    return " Desert Child - Jeu En"


@pytest.fixture()
def no_leading_white_space_text():
    return "Desert Child - Jeu En"


@pytest.fixture()
def tailing_white_space_text():
    return "Desert Child - Jeu En "


@pytest.fixture()
def no_tailing_white_space_text():
    return "Desert Child - Jeu En"


@pytest.fixture()
def outsited_tailing_and_leading_white_space_text():
    return "  Desert Child - Jeu En "


@pytest.fixture()
def no_outsited_tailing_and_leading_white_space_text():
    return "Desert Child - Jeu En"


@pytest.fixture()
def insited_tailing_and_leading_white_space_text():
    return "Desert  Child  -  Jeu En"


@pytest.fixture()
def no_insited_tailing_and_leading_white_space_text():
    return "Desert Child - Jeu En"


def test_to_lower(original_text, lower_text):
    new_text = to_lower(original_text)

    assert original_text != new_text, "Text didn't change"
    assert lower_text == new_text, "Text didn't change as expected"


def test_remove_punctuation(original_text, no_punctuation_text):
    new_text = remove_punction(no_punctuation_text)

    assert original_text != new_text, "Text didn't change"
    assert no_punctuation_text == new_text, "Text didn't change as expected"


def test_remove_html(original_text, no_html_text):
    new_text = remove_html(original_text)

    assert original_text != new_text, "Text didn't change"
    assert no_html_text == new_text, "Text didn't change as expected"


def test_remove_leading_white_space(leading_white_space_text, no_leading_white_space_text):
    new_text = remove_white_space(no_leading_white_space_text)

    assert leading_white_space_text != new_text, "Text didn't change"
    assert no_leading_white_space_text == new_text, "Text didn't change as expected"


def test_remove_tailing_white_space(tailing_white_space_text, no_tailing_white_space_text):
    new_text = remove_white_space(tailing_white_space_text)

    assert tailing_white_space_text != new_text, "Text didn't change"
    assert no_tailing_white_space_text == new_text, "Text didn't change as expected"


def test_remove_outsited_tailing_and_leading_white_space(
    outsited_tailing_and_leading_white_space_text, no_outsited_tailing_and_leading_white_space_text
):
    new_text = remove_white_space(no_outsited_tailing_and_leading_white_space_text)

    assert outsited_tailing_and_leading_white_space_text != new_text, "Text didn't change"
    assert (
        no_outsited_tailing_and_leading_white_space_text == new_text
    ), "Text didn't change as expected"


def test_remove_insited_tailing_and_leading_white_space(
    insited_tailing_and_leading_white_space_text, no_insited_tailing_and_leading_white_space_text
):
    new_text = remove_white_space(no_insited_tailing_and_leading_white_space_text)

    assert insited_tailing_and_leading_white_space_text != new_text, "Text didn't change"
    assert (
        no_insited_tailing_and_leading_white_space_text == new_text
    ), "Text didn't change as expected"


def test_remove_white_space(original_text, no_white_space_text):
    new_text = remove_white_space(no_white_space_text)

    assert original_text != new_text, "Text didn't change"
    assert no_white_space_text == new_text, "Text didn't change as expected"
