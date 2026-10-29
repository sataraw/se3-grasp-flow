# Generative Networks for Robot Grasping

**Proposal** · ADLR WS 2026/27, project idea 4

Georgii Fatakhov (Technical University of Munich) · Awar Satar (Technical University of Munich)

---

## I. Topic and Main Idea

Grasp synthesis for a parallel-jaw gripper is inherently multimodal, since a single object admits many valid grasps. Generative models address this by learning a distribution over 6-DoF end-effector poses conditioned on the observed object point cloud [1], [2]. Diffusion- and flow-based grasp generators on SE(3) [3], [4] achieve state-of-the-art quality, but they require tens of network evaluations per sample, which limits their use in time-critical robotic applications. MeanFlow [5] and its extension to Riemannian manifolds [6] reduce sampling to a few or a single network evaluation by learning the average velocity over a time interval instead of the instantaneous velocity. Two recent works apply this principle to grasping. GraspMF [7] learns an average velocity field on $SO(3) \times \mathbb{R}^3$ and matches diffusion and flow baselines on ACRONYM with at most five network evaluations, at up to 39 times lower latency. GraspMeanFlow [8] additionally preserves SE(3)-equivariance under few-step sampling and reaches with a single evaluation the distribution coverage that an iterative SE(3) flow model attains after five.

The goal of this project is to reimplement few-step, SE(3)-equivariant grasp generation, progressing from EquiGraspFlow [4] to GraspMeanFlow [8], on the ACRONYM dataset [1], and to evaluate the resulting trade-off between grasp quality and inference latency in simulation with MuJoCo. Semantically conditioned grasp generation and additional training data are considered as optional extensions.

## II. Technical Outline

**Base model.** The base model is EquiGraspFlow [4], a conditional flow-matching model on $SO(3) \times \mathbb{R}^3$ trained with Riemannian conditional flow matching [9], [10]. A VN-DGCNN encoder maps the object point cloud to an equivariant feature, and a vector-neuron head predicts the angular and linear velocity of the grasp pose. The prior is uniform on $SO(3)$ and Gaussian around the point-cloud centroid. Since the prior is SE(3)-invariant and the network is SE(3)-equivariant by construction, generated grasps transform consistently with the object, and rotation augmentation is not required in principle. We build on the public implementation and released checkpoint of EquiGraspFlow.

**Few-step model.** For few-step generation, we reimplement the semigroup variant of GraspMeanFlow [8] (GMF-SG) on top of the EquiGraspFlow codebase, as no reference implementation is publicly available. The network is extended to predict the average velocity $(\omega^\text{avg}, v^\text{avg})(s, t, T_t)$ over an interval $[s, t]$. The additional time input $h = t - s$ is injected through the existing equivariant time-lifting mechanism, which preserves equivariance and adds a negligible number of parameters. Training combines a flow-matching boundary term at $s = t$, which anchors the field to the data distribution, with a semigroup consistency term, which requires a direct step over $[s, t]$ to agree with the stop-gradient composition of two steps through an intermediate time $m \in [s, t]$. Both terms rely only on the exponential and logarithm maps of $SO(3)$ and avoid Jacobian–vector products. At inference, a sample is obtained by the update $R \leftarrow \exp(-\Delta t\,[\omega^\text{avg}])\,R$ and $x \leftarrow x - \Delta t\, v^\text{avg}$, with one network evaluation per step; a single step yields one-step generation.

**Data.** We use ACRONYM [1] with the four-category split of EquiGraspFlow (Bowl, Laptop, Mug, Pencil). Test objects are unseen instances of the training categories and are presented in random orientations.

**Evaluation.** Distribution coverage is measured by the earth mover's distance (EMD) between generated and ground-truth grasp sets. As this metric is independent of the simulator, it permits direct comparison with published results. Physical feasibility is first assessed geometrically by the collision rate, the fraction of grasps in which the gripper intersects the object, and the contact rate, the fraction of grasps in which the object lies between the fingers. It is then measured by the success rate of a lift-and-hold test in MuJoCo, which replaces the Isaac Gym protocol of prior work. The test is calibrated with ground-truth grasps, which should succeed with high probability, and with random grasps, which should fail. 

Inference cost is reported as the number of network evaluations (NFE) and the wall-clock time per sample. Both models are compared at matched NFE $\in \{1, 2, 5, 10, 20\}$ following the protocol of [8], with EquiGraspFlow’s default sampler as an additional reference point.

**Technical contribution.** The project contributes, first, an independent reimplementation of GraspMeanFlow-SG [8] from the publication alone, including a verification of the reported few-step results. Second, it provides a MuJoCo-based grasp-success evaluation that is calibrated against ground-truth and random grasps, applied identically to all methods, and independent of the deprecated Isaac Gym, which also makes it a possible starting point for a later transfer to Isaac Sim. Third, it yields a controlled comparison of flow matching and MeanFlow with identical backbone, data, and evaluation across the full range of sampling budgets. Optional extensions are the endpoint sampler of [8], part- or language-conditioned grasp generation related to constrained grasp sampling [11], and additional training data from QDGset [12].

## III. Milestones

| # | Milestone | Content | Completion criterion | Depends on |
|---|---|---|---|---|
| M1 | EquiGraspFlow baseline | Codebase setup, ACRONYM preprocessing (4 categories), evaluation of the released checkpoint, own training | EMD of own training close to the published value and to the released checkpoint | – |
| M2 | MuJoCo evaluation | Floating parallel-jaw gripper (Panda hand), convex decomposition of object meshes, collision and contact checks, lift-and-hold test, latency measurement | Ground-truth grasps succeed at ≳ 90 %, random grasps at ≈ 0 %; EquiGraspFlow evaluated across all NFE | – |
| M3 | GraspMeanFlow-SG | Two-time conditioning, boundary and semigroup loss, Euler sampler, NFE sweep | Equivariance verified within a specified numerical tolerance; success rates evaluated in MuJoCo. We aim for an EMD at NFE 1 below that of EquiGraspFlow at NFE 1 (reported: ≈ 0.42 vs. 0.73) | M1, M2 |
| M4 | Cloud scale-up | Transfer of training and evaluation to a GPU virtual machine on Google Compute Engine, with data and checkpoints in Cloud Storage; full NFE sweeps, ablations, and MuJoCo evaluation at scale | GMF-SG training on the virtual machine reproduces the local results; final experiments carried out there | M3 |
| M5 | *(optional)* Sampler improvement | Endpoint sampler with exponential $SO(3)$ schedule [8] | Implemented and evaluated against the Euler sampler. We aim for a higher success rate at NFE 5 | M3 |
| M6 | *(optional)* Extension | Part- or language-conditioned generation [11], or QDGset [12] as additional training data | Defined upon selection | M3 |

M1 and M2 are independent and are carried out in parallel. M3 constitutes the core result and requires both. Development begins locally, and M4 transfers the working pipeline to Google Cloud for the larger experiments. Should M3 remain incomplete, M1 and M2 still yield an evaluated flow-matching baseline.

## IV. Alternative Topics

As alternatives, we consider neural network-based world models (idea 11) [13] and tactile exploration of objects (idea 2) [14].

## References

[1] C. Eppner, A. Mousavian, and D. Fox, "ACRONYM: A large-scale grasp dataset based on simulation," in Proc. IEEE Int. Conf. Robotics and Automation (ICRA), 2021.

[2] D. Winkelbauer, B. Bäuml, M. Humt, N. Thuerey, and R. Triebel, "A two-stage learning architecture that generates high-quality grasps for a multi-fingered hand," in Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS), 2022.

[3] J. Urain, N. Funk, J. Peters, and G. Chalvatzaki, "SE(3)-DiffusionFields: Learning smooth cost functions for joint grasp and motion optimization through diffusion," in Proc. IEEE Int. Conf. Robotics and Automation (ICRA), 2023.

[4] B. Lim, J. Kim, J. Kim, Y. Lee, and F. C. Park, "EquiGraspFlow: SE(3)-equivariant 6-DoF grasp pose generative flows," in Proc. Conf. Robot Learning (CoRL), 2024.

[5] Z. Geng, M. Deng, X. Bai, J. Z. Kolter, and K. He, "Mean flows for one-step generative modeling," arXiv:2505.13447, 2025.

[6] D. Woo, M. Skreta, S. Park, K. Neklyudov, and S. Ahn, "Riemannian MeanFlow," in Proc. Int. Conf. Machine Learning (ICML), 2026.

[7] S. T. Bukhari, Y. Wei, R. Ni, Z. Kingston, and A. Bera, "Fast generative grasping via Lie group-constrained MeanFlow," arXiv:2608.26076, 2026.

[8] J. Kwon, Y. Bai, A. Mollaali, and G. Lin, "GraspMeanFlow: SE(3)-equivariant MeanFlow for few-step 6-DoF grasp generation," arXiv:2608.03295, 2026.

[9] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, "Flow matching for generative modeling," in Proc. Int. Conf. Learning Representations (ICLR), 2023.

[10] R. T. Q. Chen and Y. Lipman, "Flow matching on general geometries," in Proc. Int. Conf. Learning Representations (ICLR), 2024.

[11] J. Lundell, F. Verdoja, T. N. Le, A. Mousavian, D. Fox, and V. Kyrki, "Constrained generative sampling of 6-DoF grasps," in Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS), 2023.

[12] J. Huber et al., "QDGset: A large scale grasping dataset generated with quality-diversity," arXiv preprint, 2024.

[13] C. Li, A. Krause, and M. Hutter, "Robotic world model: A neural network simulator for robust policy optimization in robotics," arXiv:2501.10100, 2025.

[14] D. Watkins-Valls, J. Varley, and P. Allen, "Multi-modal geometric learning for grasping and manipulation," in Proc. IEEE Int. Conf. Robotics and Automation (ICRA), 2019.