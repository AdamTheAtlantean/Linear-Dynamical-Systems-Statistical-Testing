from manim import *
import numpy as np


config.frame_width = 9
config.frame_height = 16

TITLE = BLUE_C
ACCENT = YELLOW
MUTED = GRAY_B
SAME = GREEN_C
DIFF = RED_C
TEALISH = TEAL_C
ORANGEISH = ORANGE
PANEL = "#1f2937"
CENTER = ORIGIN


def fit_to_width(mob, margin=0.7):
    max_width = config.frame_width - margin
    if mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def label(text, font_size=30, color=WHITE):
    return fit_to_width(Text(text, font_size=font_size, color=color), margin=0.8)


def paragraph(lines, font_size=27, color=WHITE, buff=0.18, align=LEFT):
    group = VGroup(*[label(line, font_size=font_size, color=color) for line in lines])
    group.arrange(DOWN, aligned_edge=align, buff=buff)
    return group


def pill(text, color=BLUE_C, font_size=25, min_width=0, min_height=0, hpad=0.55, vpad=0.28):
    t = label(text, font_size=font_size, color=color)
    box = RoundedRectangle(
        width=max(t.width + hpad, min_width),
        height=max(t.height + vpad, min_height),
        corner_radius=0.12,
        stroke_color=color,
        fill_color=PANEL,
        fill_opacity=0.65,
    )
    return VGroup(box, t.move_to(box))


def bullet_row(symbol_tex, description, color, font_size=24):
    dot = Dot(radius=0.055, color=color)
    symbol = MathTex(symbol_tex, font_size=font_size + 5, color=color)
    text = label(description, font_size=font_size, color=WHITE)
    return VGroup(dot, symbol, text).arrange(RIGHT, buff=0.18)


def formula(tex, font_size=38, color=WHITE, margin=0.7):
    return fit_to_width(MathTex(tex, font_size=font_size, color=color), margin=margin)


def scene_title(text):
    return label(text, font_size=34, color=TITLE).to_edge(UP, buff=0.45)


class InternshipReportDraft(Scene):
    def clear(self, *extra, run_time=0.55):
        mobs = list(self.mobjects) + list(extra)
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def construct(self):
        self.camera.background_color = BLACK
        self.title_card()
        self.inverse_problem()
        self.lds_model()
        self.hypothesis_test()
        self.method_pipeline()
        self.var_estimation()
        self.isotropic_distance()
        self.memory_operator()
        self.monte_carlo_design()
        self.distance_results()
        self.parameter_sweeps()
        self.optimized_result()
        self.conclusion()

    def title_card(self):
        title = label(
            "Comparing Linear Dynamical Systems",
            font_size=43,
            color=TITLE,
        )
        subtitle = label(
            "via VAR approximation and isotropic distance",
            font_size=30,
            color=WHITE,
        )
        author = label("Adam Nolan", font_size=26, color=ACCENT)
        lab = label("LaMME / Universite Paris-Saclay", font_size=22, color=MUTED)
        stack = VGroup(title, subtitle, author, lab).arrange(DOWN, buff=0.28)
        stack.move_to(UP * 0.9)

        thesis = paragraph(
            [
                "Question:",
                "Can two observed time series tell us",
                "whether their hidden dynamical systems",
                "are the same or different?",
            ],
            font_size=28,
            color=WHITE,
            buff=0.16,
            align=CENTER,
        )
        thesis.next_to(stack, DOWN, buff=1.1)

        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP * 0.15))
        self.play(FadeIn(author), FadeIn(lab))
        self.play(LaggedStart(*[FadeIn(x, shift=UP * 0.18) for x in thesis], lag_ratio=0.12))
        self.wait(1.2)
        self.clear()

    def inverse_problem(self):
        title = scene_title("Understanding the Inverse")
        self.play(Write(title))

        left = VGroup(
            pill("Classical", BLUE_C),
            formula(r"F = \frac{dp}{dt}", font_size=40, color=BLUE_C),
            label("known law", 24, MUTED),
            Arrow(UP, DOWN, buff=0.1, color=BLUE_C),
            label("simulate trajectories", 24, WHITE),
        ).arrange(DOWN, buff=0.28)

        right = VGroup(
            pill("This work", ACCENT),
            label("observations only", 25, WHITE),
            Arrow(DOWN, UP, buff=0.1, color=ACCENT),
            label("infer comparable structure", 25, WHITE),
            label("latent matrices are hidden", 23, MUTED),
        ).arrange(DOWN, buff=0.28)

        divider = DashedLine(UP * 3.9, DOWN * 3.9, color=GRAY_D)
        panels = VGroup(left, divider, right).arrange(RIGHT, buff=0.55)
        panels.scale_to_fit_width(config.frame_width - 0.7)
        panels.move_to(DOWN * 0.15)

        examples = paragraph(
            ["Examples: macroeconomics, neuroscience, astrophysics"],
            font_size=22,
            color=MUTED,
            align=CENTER,
        ).to_edge(DOWN, buff=0.6)

        self.play(FadeIn(left, shift=RIGHT * 0.2), Create(divider))
        self.play(FadeIn(right, shift=LEFT * 0.2))
        self.play(FadeIn(examples))
        self.wait(1.4)
        self.clear()

    def lds_model(self):
        title = scene_title("Linear Dynamical System Set-Up")
        self.play(Write(title))

        eq1 = MathTex(
            r"x_{k+1}", "=", "A", r"x_k", "+", "L", r"\epsilon_k",
            font_size=43,
            color=WHITE,
        )
        eq2 = MathTex(
            r"y_k", "=", "C", r"x_k", "+", r"\epsilon_k",
            font_size=43,
            color=WHITE,
        )
        fit_to_width(eq1)
        fit_to_width(eq2)
        eqs = VGroup(eq1, eq2).arrange(DOWN, buff=0.45).move_to(UP * 3.0)

        bullets = VGroup(
            bullet_row(r"x_k", "latent state evolving in hidden space", TEALISH, 23),
            bullet_row(r"y_k", "observable time series we actually measure", BLUE_C, 23),
            bullet_row(r"A", "latent dynamics matrix", ORANGEISH, 23),
            bullet_row(r"C", "observation map from hidden state to data", PURPLE_C, 23),
            bullet_row(r"L", "noise injection / innovation loading", GREEN_C, 23),
            bullet_row(r"\epsilon_k", "independent observation noise", ACCENT, 23),
        )
        bullets.arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        bullets.scale_to_fit_width(config.frame_width - 0.9)
        bullets.move_to(DOWN * 1.0)

        note = label("Only y_k is observed; the rest must be inferred indirectly.", 22, MUTED)
        note.next_to(bullets, DOWN, buff=0.45)

        self.play(Write(eq1))
        self.play(Write(eq2))

        self.play(
            TransformFromCopy(eq1[3], bullets[0]),
            eq1[0].animate.set_color(TEALISH),
            eq1[3].animate.set_color(TEALISH),
            eq2[3].animate.set_color(TEALISH),
            run_time=1.25,
        )
        self.play(
            TransformFromCopy(eq2[0], bullets[1]),
            eq2[0].animate.set_color(BLUE_C),
            run_time=1.25,
        )
        self.play(
            TransformFromCopy(eq1[2], bullets[2]),
            eq1[2].animate.set_color(ORANGEISH),
            run_time=1.25,
        )
        self.play(
            TransformFromCopy(eq2[2], bullets[3]),
            eq2[2].animate.set_color(PURPLE_C),
            run_time=1.25,
        )
        self.play(
            TransformFromCopy(eq1[5], bullets[4]),
            eq1[5].animate.set_color(GREEN_C),
            run_time=1.25,
        )
        self.play(
            TransformFromCopy(eq1[6], bullets[5]),
            eq1[6].animate.set_color(ACCENT),
            eq2[5].animate.set_color(ACCENT),
            run_time=1.25,
        )
        self.play(FadeIn(note, shift=UP * 0.15))
        self.wait(1.5)
        self.clear()

    def hypothesis_test(self):
        title = scene_title("Same System or Different System?")
        self.play(Write(title))

        h0 = VGroup(
            pill("H0: SAME", SAME),
            formula(
                r"(A^{(1)},C^{(1)},L^{(1)}) = (A^{(2)},C^{(2)},L^{(2)})",
                font_size=29,
                color=SAME,
            ),
            paragraph(["two independent trajectories", "one underlying LDS"], 23, WHITE, align=CENTER),
        ).arrange(DOWN, buff=0.32)

        h1 = VGroup(
            pill("H1: DIFFERENT", DIFF),
            formula(
                r"(A^{(1)},C^{(1)},L^{(1)}) \ne (A^{(2)},C^{(2)},L^{(2)})",
                font_size=29,
                color=DIFF,
            ),
            paragraph(["two trajectories", "two distinct LDS models"], 23, WHITE, align=CENTER),
        ).arrange(DOWN, buff=0.32)

        h0_box = SurroundingRectangle(h0, color=SAME, buff=0.28)
        h1_box = SurroundingRectangle(h1, color=DIFF, buff=0.28)
        stack = VGroup(VGroup(h0_box, h0), VGroup(h1_box, h1)).arrange(DOWN, buff=0.65)
        stack.scale_to_fit_width(config.frame_width - 0.75)
        stack.move_to(DOWN * 0.15)

        note = label("Decision must be made from y only.", 24, ACCENT).to_edge(DOWN, buff=0.55)

        self.play(FadeIn(h0), Create(h0_box))
        self.play(FadeIn(h1), Create(h1_box))
        self.play(Write(note))
        self.wait(1.3)
        self.clear()

    def method_pipeline(self):
        title = scene_title("Method: Compare in Observable Space")
        self.play(Write(title))

        names = [
            ("Observed series", TEALISH),
            ("VAR(p) fit", BLUE_C),
            ("Coefficient vector", PURPLE_C),
            ("Isotropic distance", ACCENT),
            ("Threshold decision", ORANGEISH),
        ]
        nodes = VGroup(
            *[
                pill(name, color, font_size=26, min_width=5.4, min_height=0.72, hpad=0.9, vpad=0.36)
                for name, color in names
            ]
        )
        nodes.arrange(DOWN, buff=0.7)
        nodes.move_to(ORIGIN + UP * 0.25)

        arrows = VGroup()
        for a, b in zip(nodes[:-1], nodes[1:]):
            arrows.add(Arrow(a.get_bottom(), b.get_top(), buff=0.12, color=GRAY_B, max_tip_length_to_length_ratio=0.18))

        side = paragraph(
            [
                "Do not estimate A, C, L directly.",
                "Estimate the autoregressive footprint",
                "left in the observations.",
            ],
            font_size=23,
            color=MUTED,
            align=CENTER,
        ).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(nodes[0], shift=UP * 0.15), run_time=0.8)
        for i in range(len(arrows)):
            self.play(Create(arrows[i]), FadeIn(nodes[i + 1], shift=UP * 0.15), run_time=0.95)
        self.play(FadeIn(side))
        self.wait(1.2)
        self.clear()

    def var_estimation(self):
        title = scene_title("From LDS to VAR(p) to Least Squares")
        self.play(Write(title))

        lds = VGroup(
            formula(r"x_{k+1} = A x_k + L\epsilon_k", font_size=36),
            formula(r"y_k = Cx_k + \epsilon_k", font_size=36),
        ).arrange(DOWN, buff=0.3)
        lds_box = SurroundingRectangle(lds, color=BLUE_C, buff=0.26)
        lds_label = pill("1. LDS model", BLUE_C, font_size=24, min_width=3.0)
        lds_group = VGroup(lds_label, VGroup(lds_box, lds)).arrange(DOWN, buff=0.28)
        lds_group.move_to(UP * 4.65)

        logic = paragraph(
            [
                "Hidden state unavailable.",
                "Use observation history instead.",
            ],
            font_size=23,
            color=MUTED,
            align=CENTER,
            buff=0.12,
        ).move_to(UP * 2.95)

        infinite_ar = MathTex(
            r"y_k = \sum_{i=1}^{\infty} C",
            r"(A-LC)",
            r"^{i-1}L\,y_{k-i} + u_k",
            font_size=32,
            color=ACCENT,
        )
        fit_to_width(infinite_ar, margin=0.45)
        infinite_ar.move_to(UP * 1.7)

        f_brace = Brace(infinite_ar[1], DOWN, buff=0.08, color=ACCENT)
        f_label = MathTex(r"F", font_size=30, color=ACCENT).next_to(f_brace, DOWN, buff=0.08)

        memory_note = paragraph(
            [
                "Stable F: older terms decay.",
                "Finite VAR(p): controlled approximation.",
            ],
            font_size=21,
            color=MUTED,
            align=CENTER,
            buff=0.12,
        ).move_to(UP * 0.35)

        var = formula(
            r"y_k \approx \Phi_1 y_{k-1}+\cdots+\Phi_p y_{k-p}+u_k",
            font_size=34,
            color=WHITE,
            margin=0.45,
        ).move_to(DOWN * 1.25)

        regression = formula(r"Y = XB + U", font_size=48, color=ACCENT)
        regression.move_to(DOWN * 2.85)
        estimate = formula(r"\hat{B}=(X^T X)^{-1}X^T Y", font_size=38, color=WHITE)
        estimate.move_to(DOWN * 4.0)

        footer = paragraph(
            [
                "B contains the VAR coefficients.",
                "Least squares estimates B from data.",
            ],
            font_size=21,
            color=MUTED,
            align=CENTER,
            buff=0.12,
        )
        footer.move_to(DOWN * 5.15)

        self.play(FadeIn(lds_label, shift=UP * 0.15), Write(lds), Create(lds_box), run_time=1.6)
        self.play(FadeIn(logic, shift=UP * 0.12), run_time=1.1)
        self.play(TransformFromCopy(lds, infinite_ar), run_time=2.2)
        self.play(GrowFromCenter(f_brace), Write(f_label), run_time=1.1)
        self.play(FadeIn(memory_note, shift=UP * 0.12), run_time=1.2)
        self.play(TransformFromCopy(infinite_ar, var), run_time=2.0)
        self.play(TransformFromCopy(var, regression), run_time=1.8)
        self.play(Write(estimate), run_time=1.5)
        self.play(FadeIn(footer, shift=UP * 0.1), run_time=1.1)
        self.wait(2.0)
        self.clear()

    def isotropic_distance(self):
        title = scene_title("Isotropic Distance")
        self.play(Write(title))

        dist = formula(
            r"D = \sum_{i=1}^{p} \left\|\hat{\Phi}^{(1)}_i - \hat{\Phi}^{(2)}_i\right\|_F^2",
            font_size=39,
            color=ACCENT,
        ).move_to(UP * 3.85)

        cap = paragraph(
            ["Squared Euclidean separation", "in estimated coefficient space"],
            font_size=23,
            color=MUTED,
            align=CENTER,
        )
        cap.next_to(dist, DOWN, buff=0.45)

        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=6.0,
            y_length=4.2,
            background_line_style={"stroke_color": GRAY_D, "stroke_opacity": 0.35},
        ).move_to(DOWN * 1.55)

        p1 = Dot(plane.c2p(-1.2, 0.65), color=SAME, radius=0.08)
        p2 = Dot(plane.c2p(1.25, -0.6), color=DIFF, radius=0.08)
        line = Line(p1.get_center(), p2.get_center(), color=ACCENT)
        brace = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector(), color=ACCENT)
        d_label = MathTex(r"D", font_size=38, color=ACCENT).next_to(brace, RIGHT, buff=0.08)

        self.play(Write(dist))
        self.play(FadeIn(cap, shift=UP * 0.1))
        self.play(Create(plane))
        self.play(FadeIn(p1), FadeIn(p2))
        self.play(Create(line), GrowFromCenter(brace), Write(d_label))
        self.wait(1.3)
        self.clear()

    def memory_operator(self):
        title = scene_title("Memory Lives in F")
        self.play(Write(title))

        eq = formula(r"F = A - LC", font_size=48, color=ACCENT)
        cond = formula(r"\rho(F) < 1 \quad \Rightarrow \quad \text{decaying memory}", font_size=33)
        context = paragraph(
            [
                "F controls how strongly past observations persist.",
                "rho(F) near 1 means a longer memory tail.",
            ],
            font_size=22,
            color=MUTED,
            align=CENTER,
            buff=0.13,
        )
        header_group = VGroup(eq, cond, context).arrange(DOWN, buff=0.28).move_to(UP * 3.35)

        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5.4,
            y_length=5.4,
            background_line_style={"stroke_color": GRAY_D, "stroke_opacity": 0.28},
        ).move_to(DOWN * 1.75)

        x0 = np.array([2.3, 1.8])
        short_matrix = np.array([[0.62, 0.04], [-0.02, 0.58]])
        long_matrix = np.array([[0.94, -0.05], [0.04, 0.88]])
        short_dot = Dot(plane.c2p(*x0), color=TEALISH)
        long_dot = Dot(plane.c2p(*x0), color=ORANGEISH)
        short_path = TracedPath(short_dot.get_center, stroke_color=TEALISH, stroke_width=4)
        long_path = TracedPath(long_dot.get_center, stroke_color=ORANGEISH, stroke_width=4)

        labels = VGroup(
            label("short memory", 22, TEALISH),
            label("long memory", 22, ORANGEISH),
        ).arrange(RIGHT, buff=0.6).to_edge(DOWN, buff=0.45)

        self.play(Write(eq), Write(cond))
        self.play(FadeIn(context, shift=UP * 0.12))
        self.play(Create(plane), FadeIn(labels))
        self.add(short_path, long_path, short_dot, long_dot)

        xs = x0.copy()
        xl = x0.copy()
        for _ in range(11):
            xs = short_matrix @ xs
            xl = long_matrix @ xl
            self.play(
                short_dot.animate.move_to(plane.c2p(*xs)),
                long_dot.animate.move_to(plane.c2p(*xl)),
                run_time=0.18,
            )

        self.wait(1.1)
        self.clear()

    def monte_carlo_design(self):
        title = scene_title("Monte Carlo Experiment")
        self.play(Write(title))

        outer = VGroup(
            pill("Outer loop: sample systems", BLUE_C),
            label("n realizations", 24, MUTED),
        ).arrange(DOWN, buff=0.18)
        inner = VGroup(
            pill("Inner loop: simulate trajectories", TEALISH),
            label("m trials per realization", 24, MUTED),
        ).arrange(DOWN, buff=0.18)
        compare = VGroup(
            pill("Build H0 and H1 distance distributions", ACCENT),
            label("then choose threshold tau", 24, MUTED),
        ).arrange(DOWN, buff=0.18)

        stack = VGroup(outer, inner, compare).arrange(DOWN, buff=0.7)
        stack.scale_to_fit_width(config.frame_width - 0.8)
        stack.move_to(UP * 0.15)

        arrows = VGroup()
        for a, b in zip(stack[:-1], stack[1:]):
            arrows.add(Arrow(a.get_bottom(), b.get_top(), color=GRAY_B, buff=0.18))

        self.play(FadeIn(outer, shift=UP * 0.15))
        self.play(Create(arrows[0]), FadeIn(inner, shift=UP * 0.15))
        self.play(Create(arrows[1]), FadeIn(compare, shift=UP * 0.15))
        self.wait(1.4)
        self.clear()

    def distance_results(self):
        title = scene_title("Result: Distances Separate")
        self.play(Write(title))

        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1.0, 0.25],
            x_length=7.0,
            y_length=4.7,
            tips=False,
            axis_config={"color": GRAY_B},
        ).move_to(UP * 0.65)

        same_curve = axes.plot(lambda x: 0.95 * np.exp(-((x - 1.7) ** 2) / 0.7), x_range=[0, 4.2], color=SAME)
        diff_curve = axes.plot(lambda x: 0.78 * np.exp(-((x - 6.8) ** 2) / 1.1), x_range=[3.5, 10], color=DIFF)
        tau_tracker = ValueTracker(7.5)
        tau_line = always_redraw(
            lambda: DashedLine(
                axes.c2p(tau_tracker.get_value(), 0),
                axes.c2p(tau_tracker.get_value(), 0.92),
                color=ACCENT,
            )
        )
        tau = always_redraw(
            lambda: MathTex(r"\tau", font_size=36, color=ACCENT).next_to(tau_line, UP, buff=0.08)
        )
        target = always_redraw(
            lambda: label("candidate threshold", 23, ACCENT).next_to(tau_line, DOWN, buff=0.36)
        )

        h0_label = VGroup(
            MathTex(r"H_{0}", font_size=31, color=SAME),
            label("near zero", 23, SAME),
        ).arrange(RIGHT, buff=0.12).move_to(axes.c2p(1.7, 1.08))
        h1_label = VGroup(
            MathTex(r"H_{1}", font_size=31, color=DIFF),
            label("shifted right", 23, DIFF),
        ).arrange(RIGHT, buff=0.12).move_to(axes.c2p(6.8, 0.93))
        labels = VGroup(h0_label, h1_label)

        bottom = paragraph(
            [
                "Baseline: H0 concentrates near zero.",
                "H1 moves away and spreads out.",
                "We sweep tau, then choose the value",
                "that best separates the two distributions.",
            ],
            font_size=22,
            color=WHITE,
        ).to_edge(DOWN, buff=0.55)

        self.play(Create(axes))
        self.play(Create(same_curve), FadeIn(labels[0]))
        self.play(Create(diff_curve), FadeIn(labels[1]))
        self.play(FadeIn(tau_line), Write(tau), FadeIn(target))
        self.play(tau_tracker.animate.set_value(2.9), run_time=2.0, rate_func=smooth)
        self.play(tau_tracker.animate.set_value(4.4), run_time=1.2, rate_func=smooth)
        self.play(FadeIn(bottom))
        self.wait(1.5)
        self.clear()

    def parameter_sweeps(self):
        title = scene_title("What Controls Performance?")
        self.play(Write(title))

        rows = [
            ("Effective sample size T", "AUC rises quickly", SAME),
            ("VAR order p", "harder as p grows", ORANGEISH),
            ("Observation dimension d_y", "more coefficients to estimate", ORANGEISH),
            ("Latent dimension d_x", "trade-off: diversity vs complexity", BLUE_C),
            ("Regime separation delta", "larger gap improves separation", SAME),
            ("Noise scale sigma", "performance stays robust", TEALISH),
        ]

        table = VGroup()
        for left, right, color in rows:
            l = label(left, 22, WHITE)
            r = label(right, 22, color)
            row = VGroup(l, r).arrange(RIGHT, buff=0.45)
            row.scale_to_fit_width(config.frame_width - 0.9)
            table.add(row)
        table.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(DOWN * 0.05)

        rule = label(
            "Core trade-off: dynamical separability vs estimation variability",
            23,
            ACCENT,
        ).to_edge(DOWN, buff=0.45)

        self.play(LaggedStart(*[FadeIn(row, shift=RIGHT * 0.15) for row in table], lag_ratio=0.12))
        self.play(Write(rule))
        self.wait(1.6)
        self.clear()

    def optimized_result(self):
        title = scene_title("Under Favorable Conditions")
        self.play(Write(title))

        score = label("Perfect separation on the sampled dataset", 35, ACCENT)
        score.move_to(UP * 2.55)

        metrics = VGroup(
            pill("false positives: 0", SAME, 24),
            pill("false negatives: 0", SAME, 24),
            pill("near-perfect AUC / F1", BLUE_C, 24),
        ).arrange(DOWN, buff=0.35).next_to(score, DOWN, buff=0.7)

        caveat = paragraph(
            [
                "This happens when sample size is sufficient,",
                "model complexity is moderate,",
                "and dynamical regimes are clearly separated.",
            ],
            font_size=24,
            color=WHITE,
            align=CENTER,
        ).to_edge(DOWN, buff=0.65)

        self.play(Write(score))
        self.play(LaggedStart(*[FadeIn(x, shift=UP * 0.12) for x in metrics], lag_ratio=0.18))
        self.play(FadeIn(caveat))
        self.wait(1.5)
        self.clear()

    def conclusion(self):
        title = scene_title("Takeaway")
        self.play(Write(title))

        points = paragraph(
            [
                "Latent systems can be compared",
                "through their observable VAR footprints.",
                "",
                "The isotropic metric is simple:",
                "Euclidean energy in coefficient space.",
                "",
                "The next theoretical target:",
                "derive tau without simulation.",
            ],
            font_size=27,
            color=WHITE,
            align=CENTER,
            buff=0.2,
        ).move_to(UP * 0.15)

        final = label("From hidden dynamics to a one-dimensional test.", 25, ACCENT)
        final.to_edge(DOWN, buff=0.7)

        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.13) for p in points], lag_ratio=0.08))
        self.play(Write(final))
        self.wait(2.0)
