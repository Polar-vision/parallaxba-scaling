# Suggested reviewers — IJRR submission

Verified 4 November 2026 against public institutional pages. Enter these in ScholarOne;
the cover letter only points to the system, it does not list them.

---

## 1. Dr Liang Zhao

| field | value |
|---|---|
| E-mail | Liang.Zhao@ed.ac.uk |
| Institution | The University of Edinburgh |
| Department | Institute of Perception, Action and Behaviour, School of Informatics |
| Country | United Kingdom |
| Position | Reader in Robot Systems |

First author of ParallaxBA (Zhao et al., 2015). He wrote the original scaled-ray
formulation, so he is the person best placed to check that the anchor-dependent chart
`A_q = A + J_X G` really is a reparameterization of *his* equations and not of a variant
of them. Also holds a PhD from the Institute of Remote Sensing and GIS, Peking University,
so he reads the photogrammetric side of the argument as well as the robotics side.

Reason to enter: *First author of the ParallaxBA formulation examined in this paper;
expert in BA parametrization and SLAM optimization.*

## 2. Prof Gamini Dissanayake

| field | value |
|---|---|
| E-mail | Gamini.Dissanayake@uts.edu.au |
| Institution | University of Technology Sydney |
| Department | School of Electrical, Mechanical and Biomedical Engineering |
| Country | Australia |
| Position | Emeritus Professor (retired December 2020, still publishing) |

Co-author of ParallaxBA and of the original SLAM literature the paper builds on. His
interest is estimation rather than parametrization detail, which is the right vantage point
for the claim that the chart changes the damping metric and nothing about the information.

Reason to enter: *Co-author of ParallaxBA; estimation and SLAM, able to judge the
preconditioning-versus-information distinction.*

## 3. Dr Manolis Lourakis

| field | value |
|---|---|
| E-mail | lourakis@ics.forth.gr |
| Institution | Foundation for Research and Technology – Hellas (FORTH) |
| Department | Institute of Computer Science, Computational Vision and Robotics Laboratory |
| Country | Greece |
| Position | Research Director |

Author of SBA, cited in this paper. He implemented the Schur elimination that the theorem
is about, so he can judge whether "the matrix the solver factorizes is unchanged" is
stated in a way that matches what an implementation actually assembles and factorizes.
Not connected to ParallaxBA and not a co-author of any earlier work by the author.

Reason to enter: *Author of SBA; expert in sparse BA implementation and numerical
optimization for structure-from-motion.*

---

## Who was deliberately left out, and why

**Prof Lei Yan and Dr Yanbiao Sun (Peking University).** Both are ParallaxBA co-authors
and both are co-authors of Zuo et al. (2023), the study this paper revises. Lei Yan has also
been the corresponding author on much of the author's earlier work. Suggesting either would
be an immediate conflict.

**Prof Shoudong Huang (UTS).** ParallaxBA co-author and therefore a natural name, but he
has served as an Associate Editor of IJRR (listed 2024–present on his own page; the UTS
profile dates the term 2024–2025, so it may have lapsed). Avoid: proposing a serving editor
reads either as naivety or as an attempt to route the paper. If the term has in fact ended
and you want him on the list, he is otherwise the strongest single candidate — he knows the
derivation and he is not a co-author of the 2023 study.

That exhausts the ParallaxBA author list: five authors, two conflicted by co-authorship,
one excluded as an editor of the target journal. The third slot has to come from outside,
which is why Lourakis is proposed.

## A risk worth naming before you submit

Reviewers 1 and 2 wrote the method the paper partly walks back. The paper's verdict is not
hostile — it says the chart buys a real preconditioning and a real trust-region geometry —
but it does reduce the claimed benefit from "more information" to "better scaling", and the
authors of the original are the people most likely to resist that reduction.

Two ways to read this, and both are defensible:

- **Send it to them.** They are the only people who can find a genuine error in the
  derivation, and a favourable report from them is worth more than three from elsewhere.
- **Do not.** A defensive report from the method's authors is hard to answer, because the
  disagreement is about what their own result means.

If you want the middle path, keep Zhao (he is the one who can check the algebra) and replace
Dissanayake with a second independent name. Say the word and I will verify one.
