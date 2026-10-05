"""
Tests for text preprocessing and normalization in VitaMap ResumeForge.
Uses standard library unittest for zero-dependency reliability.
"""

import unittest
from backend.src.preprocessing import (
    clean_text_minimal,
    clean_text_conservative,
    clean_text_stopwords,
    preprocess_text,
    extract_text_from_html,
)


class TestPreprocessing(unittest.TestCase):

    def test_normal_resume_text(self):
        sample = "Experienced Senior Accountant with over 8 years in financial reporting and tax preparation."
        cleaned = clean_text_minimal(sample)
        self.assertIn("Experienced Senior Accountant", cleaned)
        self.assertIn("financial reporting", cleaned)

    def test_html_entities_and_tags(self):
        html_sample = "<div><h3>Skills &amp; Competencies</h3><p>Financial analysis &bull; Budgeting</p></div>"
        cleaned = clean_text_minimal(html_sample)
        self.assertNotIn("<", cleaned)
        self.assertNotIn(">", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertIn("Skills & Competencies", cleaned)
        self.assertIn("Financial analysis", cleaned)

    def test_repeated_whitespace_and_newlines(self):
        messy = "Summary:\n\n\n\nExperienced developer.\t\tSkills:   Python   and    SQL."
        cleaned = clean_text_minimal(messy)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\t", cleaned)
        self.assertIn("Experienced developer.", cleaned)
        self.assertIn("Python and SQL.", cleaned)

    def test_technical_tokens_preservation_variant_a(self):
        tech_text = "Proficient in C++, C#, .NET Framework, SQL, Python, Java, AWS, Azure, GCP, and TensorFlow."
        cleaned_a = clean_text_minimal(tech_text)
        self.assertIn("C++", cleaned_a)
        self.assertIn("C#", cleaned_a)
        self.assertIn(".NET", cleaned_a)
        self.assertIn("SQL", cleaned_a)
        self.assertIn("Python", cleaned_a)
        self.assertIn("Java", cleaned_a)
        self.assertIn("AWS", cleaned_a)
        self.assertIn("TensorFlow", cleaned_a)

    def test_technical_tokens_preservation_variant_b(self):
        tech_text = "Skills: C++, C#, and .NET Framework with Node.js and SQL."
        cleaned_b = clean_text_conservative(tech_text)
        self.assertIn("c++", cleaned_b)
        self.assertIn("c#", cleaned_b)
        self.assertIn(".net", cleaned_b)
        self.assertIn("node.js", cleaned_b)
        self.assertIn("sql", cleaned_b)

    def test_technical_tokens_preservation_variant_c(self):
        tech_text = "Experienced in C++, C#, and .NET Framework."
        cleaned_c = clean_text_stopwords(tech_text)
        self.assertIn("c++", cleaned_c)
        self.assertIn("c#", cleaned_c)
        self.assertIn(".net", cleaned_c)
        tokens = cleaned_c.split()
        self.assertNotIn("and", tokens)
        self.assertNotIn("in", tokens)

    def test_unicode_normalization_and_ligatures(self):
        unicode_sample = "Pro\ufb01cient in ﬁnancial modeling – ‘Budget 2024’."
        cleaned = clean_text_minimal(unicode_sample)
        self.assertNotIn("\ufb01", cleaned)
        self.assertIn("Proficient", cleaned)
        self.assertIn("financial", cleaned)
        self.assertIn("-", cleaned)
        self.assertIn("'Budget 2024'", cleaned)

    def test_bullet_point_cleaning(self):
        bullet_sample = "Highlights: • Team Leadership ▪ Project Delivery ◆ Risk Mitigation ● Agile"
        cleaned = clean_text_minimal(bullet_sample)
        self.assertNotIn("•", cleaned)
        self.assertNotIn("▪", cleaned)
        self.assertNotIn("◆", cleaned)
        self.assertNotIn("●", cleaned)
        self.assertIn("Team Leadership", cleaned)
        self.assertIn("Project Delivery", cleaned)

    def test_empty_and_none_input(self):
        self.assertEqual(clean_text_minimal(""), "")
        self.assertEqual(clean_text_minimal(None), "")
        self.assertEqual(clean_text_minimal("   \n\t   "), "")
        self.assertEqual(clean_text_conservative(""), "")
        self.assertEqual(clean_text_conservative(None), "")
        self.assertEqual(clean_text_stopwords(""), "")
        self.assertEqual(clean_text_stopwords(None), "")

    def test_joined_word_safety(self):
        text = "Audio Engineer proficient in Ableton Live, Studio One, and Pro Tools."
        cleaned = clean_text_minimal(text)
        self.assertIn("Ableton Live", cleaned)

    def test_preprocess_text_dispatcher(self):
        sample = "Developer with C++ &amp; Python skills."
        var_a = preprocess_text(sample, variant="A")
        var_b = preprocess_text(sample, variant="B")
        var_c = preprocess_text(sample, variant="C")

        self.assertIn("C++ & Python", var_a)
        self.assertIn("c++", var_b)
        self.assertIn("c++", var_c)
        self.assertNotIn("with", var_c.split())

        with self.assertRaises(ValueError):
            preprocess_text(sample, variant="INVALID")


if __name__ == "__main__":
    unittest.main()
