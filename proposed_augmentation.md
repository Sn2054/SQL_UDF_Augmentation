# Implementation Specification: Semantic Graph Coarsening and Refinement Augmentor for GRACEFUL

<!-- Input query and UDF
        ↓
GRACEFUL graph construction
        ↓
Node feature encoding
        ↓
Graph Augmentor
    ├── Region extraction
    ├── Coarsening
    ├── Coarse message passing
    └── Refinement
        ↓
Original GRACEFUL message passing
        ↓
Runtime prediction -->

## Overall Pipeline with Equations

The complete processing pipeline is:

```text
Input query and UDF
        ↓
GRACEFUL graph construction
        ↓
Node feature encoding
        ↓
Graph Augmentor
    ├── Region extraction
    ├── Coarsening
    ├── Coarse message passing
    └── Refinement
        ↓
Original GRACEFUL message passing
        ↓
Runtime prediction
```

### 1. GRACEFUL Graph Construction

The input query plan and UDF are converted into a graph:

$$
G=(V,E,X)
$$

Equivalently:

$$
G=(A,X)
$$

where:

- $V$ is the node set.
- $E$ is the edge set.
- $A$ is the adjacency matrix.
- $X$ is the raw node-feature matrix.

The adjacency matrix is defined as:

$$
A_{ij}
=
\begin{cases}
1, & \text{if information flows from node } i \text{ to node } j,\\
0, & \text{otherwise.}
\end{cases}
$$

---

### 2. Node Feature Encoding

The raw node features are converted into hidden representations:

$$
H^{(0)}
=
\operatorname{Encoder}(X)
$$

For each node $i$:

$$
h_i^{(0)}
=
\operatorname{Encoder}_{\tau_i}(x_i)
$$

where:

- $x_i$ is the raw feature vector of node $i$.
- $\tau_i$ is the node type.
- $h_i^{(0)}$ is the encoded node representation.

The encoded graph is:

$$
G^{(0)}
=
\left(A,H^{(0)}\right)
$$

---

## Graph Augmentor

The graph augmentor receives:

$$
G^{(0)}
=
\left(A,H^{(0)}\right)
$$

and produces refined original-node features:

$$
H'
=
\operatorname{GraphAugmentor}
\left(A,H^{(0)}\right)
$$

The graph augmentor contains four stages:

```text
Region extraction
        ↓
Coarsening
        ↓
Coarse message passing
        ↓
Refinement
```

---

### 3. Region Extraction

Region extraction identifies loop and branch regions in the original graph:

$$
\mathcal{R},C
=
\operatorname{RegionExtraction}(A)
$$

where:

$$
\mathcal{R}
=
\left\{
R_1,R_2,\ldots,R_M
\right\}
$$

is the set of detected regions, and:

$$
C\in\{0,1\}^{N\times M}
$$

is the node-to-region membership matrix.

The membership matrix is defined as:

$$
C_{ir}
=
\begin{cases}
1, & \text{if original node } i \text{ belongs to region } R_r,\\
0, & \text{otherwise.}
\end{cases}
$$

Here:

- $N$ is the number of original nodes.
- $M$ is the number of detected loop and branch regions.
- $C_{ir}=1$ means that node $i$ belongs to region $R_r$.

Region extraction does not change the node representations. It only identifies which original nodes belong to each loop or branch region.

---

### 4. Coarsening

Coarsening creates one supernode for every detected loop or branch region.

The coarse adjacency matrix is:

$$
A_c
=
C^\top A C
$$

The coarse node features are obtained by aggregating the representations of the original nodes belonging to each region.

#### Mean Pooling

For region $R_r$, mean pooling is:

$$
h_r^c
=
\frac{1}{|R_r|}
\sum_{i\in R_r}
h_i^{(0)}
$$

where $|R_r|$ is the number of nodes in region $R_r$.

#### Sum Pooling

Sum pooling is:

$$
h_r^c
=
\sum_{i\in R_r}
h_i^{(0)}
$$

#### Weighted Mean Pooling

Weighted mean pooling is:

$$
h_r^c
=
\frac{
\sum_{i\in R_r}
w_i h_i^{(0)}
}{
\sum_{i\in R_r}
w_i+\epsilon
}
$$

where:

- $w_i$ is the importance or execution weight of node $i$.
- $\epsilon$ is a small constant used to avoid division by zero.

The weight $w_i$ may depend on incoming rows, execution count, loop iterations, or a learned importance score.

#### Attention Pooling

The attention score for node $i$ in region $R_r$ is:

$$
\alpha_{ir}
=
\frac{
\exp(e_{ir})
}{
\sum_{j\in R_r}\exp(e_{jr})
}
$$

The region representation is:

$$
h_r^c
=
\sum_{i\in R_r}
\alpha_{ir}W_vh_i^{(0)}
$$

where:

- $e_{ir}$ is the learned attention score.
- $\alpha_{ir}$ is the normalized importance of node $i$.
- $W_v$ is a learnable projection matrix.

The initial supernode representation is:

$$
z_r^{(0)}
=
\phi_{\tau(r)}
\left(
h_r^c
\Vert
m_r
\right)
$$

where:

- $\tau(r)$ is the region type, such as loop or branch.
- $m_r$ contains region-specific metadata.
- $\Vert$ denotes concatenation.
- $\phi_{\tau(r)}$ is a region-type-specific neural network.

#### Loop Supernode

For loop region $l$, the loop supernode representation is:

$$
z_l^{(0)}
=
\phi_L
\left(
h_{\mathrm{LOOP}}
\Vert
h_{\mathrm{LOOP\_END}}
\Vert
h_l^c
\Vert
\log(1+\mathrm{nr\_iter}_l)
\Vert
\log(1+\mathrm{in\_rows}_l)
\right)
$$

where:

- $h_{\mathrm{LOOP}}$ is the loop-start representation.
- $h_{\mathrm{LOOP\_END}}$ is the loop-end representation.
- $h_l^c$ is the pooled loop-body representation.
- $\mathrm{nr\_iter}_l$ is the loop iteration count.
- $\mathrm{in\_rows}_l$ is the number of rows entering the loop.
- $\phi_L$ is the loop-specific projection network.

#### Branch Supernode

For branch region $b$, the true and false paths are pooled separately:

$$
h_b^{T}
=
\operatorname{Pool}
\left(
\left\{
h_i^{(0)}:i\in R_b^{T}
\right\}
\right)
$$

$$
h_b^{F}
=
\operatorname{Pool}
\left(
\left\{
h_i^{(0)}:i\in R_b^{F}
\right\}
\right)
$$

The branch supernode representation is:

$$
z_b^{(0)}
=
\phi_B
\left(
h_{\mathrm{BRANCH}}
\Vert
h_b^{T}
\Vert
h_b^{F}
\Vert
p_b^{T}
\Vert
p_b^{F}
\Vert
\log(1+\mathrm{in\_rows}_b)
\right)
$$

where:

- $h_{\mathrm{BRANCH}}$ is the branch-condition representation.
- $h_b^{T}$ is the true-path representation.
- $h_b^{F}$ is the false-path representation.
- $p_b^{T}$ is the true-path hit ratio.
- $p_b^{F}$ is the false-path hit ratio.
- $\mathrm{in\_rows}_b$ is the number of rows entering the branch.
- $\phi_B$ is the branch-specific projection network.

The complete initial supernode feature matrix is:

$$
Z^{(0)}
=
\begin{bmatrix}
z_1^{(0)}\\
z_2^{(0)}\\
\vdots\\
z_M^{(0)}
\end{bmatrix}
$$

Therefore, the coarse graph is:

$$
G_c
=
\left(
A_c,Z^{(0)}
\right)
$$

---

### 5. Coarse Message Passing

Coarse message passing exchanges information between connected loop and branch supernodes.

The updated supernode features are:

$$
Z^{(1)}
=
\operatorname{GNN}_c
\left(
A_c,Z^{(0)}
\right)
$$

For supernode $r$, the incoming coarse message is:

$$
m_r^c
=
\operatorname{Aggregate}
\left(
\left\{
W_{\rho(s,r)}z_s^{(0)}
:
s\in\mathcal{N}_c(r)
\right\}
\right)
$$

where:

- $\mathcal{N}_c(r)$ is the set of neighboring supernodes of region $r$.
- $\rho(s,r)$ is the relationship type between supernodes $s$ and $r$.
- $W_{\rho(s,r)}$ is a relation-specific projection matrix.
- $m_r^c$ is the aggregated coarse message.

The updated supernode representation is:

$$
z_r^{(1)}
=
\operatorname{Update}
\left(
z_r^{(0)},m_r^c
\right)
$$

A residual update can be written as:

$$
z_r^{(1)}
=
\operatorname{LayerNorm}
\left(
z_r^{(0)}
+
\operatorname{MLP}
\left(
z_r^{(0)}
\Vert
m_r^c
\right)
\right)
$$

For example, when a loop is located inside a branch path, the loop supernode sends its complete loop representation to the branch supernode:

```text
LOOP_SUPER
      ↓
BRANCH_SUPER
```

The branch supernode can therefore learn that one of its paths contains an expensive or frequently executed loop.

---

### 6. Refinement

Refinement transfers the updated supernode information back to the original nodes.

The basic coarse-context matrix is:

$$
H_{\mathrm{context}}
=
CZ^{(1)}
$$

Here, every row of $H_{\mathrm{context}}$ contains the coarse semantic context assigned to one original node.

#### Residual-Sum Refinement

Residual-sum refinement is:

$$
H'
=
H^{(0)}
+
W_cH_{\mathrm{context}}
$$

This method adds projected coarse context to the original node representation.

#### Concatenation-Based Refinement

Concatenation-based refinement is:

$$
H'
=
\operatorname{MLP}
\left(
H^{(0)}
\Vert
H_{\mathrm{context}}
\right)
$$

This method concatenates the original representation and the coarse context before passing them through an MLP.

#### Gated Residual Refinement

For node $i$, its semantic context is:

$$
c_i
=
\operatorname{Aggregate}
\left(
\left\{
z_r^{(1)}
:
C_{ir}=1
\right\}
\right)
$$

The refinement gate is:

$$
g_i
=
\sigma
\left(
W_g
\left[
h_i^{(0)}
\Vert
c_i
\right]
+b_g
\right)
$$

where:

- $c_i$ is the aggregated loop and branch context for node $i$.
- $g_i$ controls how much semantic context is added.
- $\sigma$ is the sigmoid function.
- $W_g$ and $b_g$ are learnable parameters.

The refined node representation is:

$$
h_i'
=
\operatorname{LayerNorm}
\left(
h_i^{(0)}
+
g_i\odot W_cc_i
\right)
$$

where $\odot$ denotes elementwise multiplication.

The complete refined feature matrix is:

$$
H'
=
\begin{bmatrix}
h_1'\\
h_2'\\
\vdots\\
h_N'
\end{bmatrix}
$$

The refined graph is:

$$
G'
=
(A,H')
$$

The original adjacency matrix $A$ remains unchanged. Only the original node representations are enriched with loop-level and branch-level context.

---

### 7. Original GRACEFUL Message Passing

The refined graph is passed through the original GRACEFUL message-passing network:

$$
H^{G}
=
\operatorname{GRACEFUL\_GNN}
\left(
A,H'
\right)
$$

For each original node $i$, the incoming GRACEFUL message is:

$$
m_i^{G}
=
\operatorname{Aggregate}
\left(
\left\{
W_{\rho(j,i)}h_j'
:
j\in\mathcal{N}(i)
\right\}
\right)
$$

where:

- $\mathcal{N}(i)$ is the set of neighboring nodes of node $i$.
- $\rho(j,i)$ is the edge type between nodes $j$ and $i$.
- $W_{\rho(j,i)}$ is an edge-type-specific projection matrix.
- $m_i^{G}$ is the aggregated GRACEFUL message.

The updated node representation is:

$$
h_i^{G}
=
\operatorname{Update}
\left(
h_i',m_i^{G}
\right)
$$

Messages propagate through:

```text
Refined UDF nodes
        ↓
RET node
        ↓
Query-plan operators
        ↓
Root query node
```

The final graph representation is obtained from the root query node:

$$
h_{\mathrm{root}}
=
H_{\mathrm{root}}^{G}
$$

---

### 8. Runtime Prediction

The runtime is predicted using the original GRACEFUL regression head:

$$
\widehat{y}
=
\operatorname{Regressor}
\left(
h_{\mathrm{root}}
\right)
$$

Equivalently:

$$
\widehat{y}
=
\operatorname{GRACEFUL}
\left(
A,H'
\right)
$$

---

## Complete Mathematical Pipeline

The original graph is:

$$
G=(A,X)
$$

The node features are encoded as:

$$
H^{(0)}
=
\operatorname{Encoder}(X)
$$

Loop and branch regions are extracted:

$$
\mathcal{R},C
=
\operatorname{RegionExtraction}(A)
$$

The coarse adjacency matrix is created:

$$
A_c
=
C^\top A C
$$

The coarse supernode representations are created:

$$
Z^{(0)}
=
\operatorname{Coarsen}
\left(
C,H^{(0)}
\right)
$$

Coarse message passing updates the supernodes:

$$
Z^{(1)}
=
\operatorname{GNN}_c
\left(
A_c,Z^{(0)}
\right)
$$

The supernode information is refined back to the original nodes:

$$
H'
=
\operatorname{Refine}
\left(
H^{(0)},C,Z^{(1)}
\right)
$$

The refined graph is processed by the original GRACEFUL message-passing network:

$$
H^{G}
=
\operatorname{GRACEFUL\_GNN}
\left(
A,H'
\right)
$$

Finally, the runtime is predicted:

$$
\widehat{y}
=
\operatorname{Regressor}
\left(
H_{\mathrm{root}}^{G}
\right)
$$

The graph augmentor does not replace the original GRACEFUL graph or runtime-prediction model. It creates semantic loop and branch representations and transfers this information back to the original nodes before the normal GRACEFUL message-passing process.