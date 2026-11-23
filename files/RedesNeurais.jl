### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 10000000-0000-4000-8000-000000000002
begin
    using LinearAlgebra
    using Random
    using Statistics
    using Printf

    using Flux
    import Zygote
    import Optimisers
    using MLUtils: DataLoader, splitobs
    using OneHotArrays: onehotbatch, onecold
    using Plots
    import PlutoUI
end

# ╔═╡ 10000000-0000-4000-8000-000000000001
md"""
# 🧠 Redes Neurais em Julia (`Flux.jl`): Do Básico ao Avançado
### Fundamentos Matemáticos, Diferenciação Automática, Transformers e PLN em CPU

- **Nível 1:** Álgebra Tensorial, Layout *Column-Major*, Localidade de Cache e Precisão `Float32`.
- **Nível 2:** Diferenciação Automática de Modo Reverso (*Pullbacks* no `Zygote.jl` vs Gradientes Analíticos).
- **Nível 2.5:** Funções de Ativação, Derivadas Automáticas e Regressão Logística.
- **Nível 3:** Teorema da Aproximação Universal, Viés-Variância e Regressão Não-Linear Reativa.
- **Nível 4:** Perceptrons Multicamadas (MLP), Truque *Log-Sum-Exp*, `BatchNorm`, `Dropout` e `AdamW`.
- **Nível 5:** Convoluções 2D (CNN), Conexões Residuais (*ResNet*), *ClipNorm* e *Cosine Annealing*.
- **Nível 6:** Aprendizado Não-Supervisionado — Autoencoders Não-Lineares vs PCA.
- **Nível 7:** Modelagem Sequencial, BPTT e Previsão de Séries Temporais (`RNN` vs `LSTM`).
- **Nível 8:** Mecanismos de Atenção Causal, Codificação Posicional, Transformers e *Induction Heads*.
- **Nível 9:** PLN e Construção Passo a Passo de uma SLM no Estilo de Machado de Assis (Pré-Treino + SFT).
"""

# ╔═╡ 10000000-0000-4000-8000-000000000003
begin
    BLAS.set_num_threads(max(1, div(Threads.nthreads(), 2)))
    md"""
    ### 💻 Configuração de Paralelismo na CPU
    - **Threads ativas no Julia:** `$(Threads.nthreads())` *(convoluções e laços paralelos)*
    - **Threads ativas no BLAS:** `$(BLAS.get_num_threads())` *(multiplicações matriciais `GEMM` e cabeças de atenção)*
    """
end

# ╔═╡ 10000000-0000-4000-8000-000000000004
md"""
---
## Nível 1: Álgebra Tensorial, Layout *Column-Major* e `Float32`

Em Julia, matrizes são armazenadas **por colunas** (*Column-Major*): para ``\mathbf{A} \in \mathbb{R}^{M \times N}``, o deslocamento linear na memória é ``\text{offset}(i,j) = i + (j-1)M``. Assim, no `Flux.jl` cada coluna é uma observação (``\mathbf{X} \in \mathbb{R}^{D \times N}``), mantendo os atributos contíguos na cache L1/L2 e dobrando o rendimento vetorial SIMD com `Float32`.
"""

# ╔═╡ 10000000-0000-4000-8000-000000000005
let
    Random.seed!(42)
    A = randn(Float32, 1024, 1024)
    B = randn(Float32, 1024, 1024)
    _ = sum(A[:, begin]); _ = sum(A[begin, :])

    t_col = @elapsed begin
        s1 = 0.0f0
        for j in axes(A, 2), i in axes(A, 1)
            @inbounds s1 += A[i, j]
        end
    end
    t_lin = @elapsed begin
        s2 = 0.0f0
        for i in axes(A, 1), j in axes(A, 2)
            @inbounds s2 += A[i, j]
        end
    end
    C_ativ = relu.(A * B .+ 0.5f0)

    md"""
    - **Varredura por colunas (contígua):** `$(round(t_col * 1000, digits=3)) ms` vs **por linhas:** `$(round(t_lin * 1000, digits=3)) ms`
    - **Saída fundida `relu.(A * B .+ 0.5f0)`:** `$(typeof(C_ativ))`, média `$(round(mean(C_ativ), digits=4))`
    """
end

# ╔═╡ 10000000-0000-4000-8000-000000000006
md"""
---
## Nível 2: Diferenciação Automática de Modo Reverso (`Zygote.jl`)

Na diferenciação reversa (*Backpropagation*), cada operação primitiva ``\mathbf{y} = f(\mathbf{u})`` define um operador linear adjunto (*Pullback*):

``\bar{\mathbf{u}} = \mathcal{B}_f^{\mathbf{u}}(\bar{\mathbf{y}}) = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{u}}\right)^\top \bar{\mathbf{y}}``

Isso permite obter todo o gradiente ``\nabla_{\boldsymbol{\theta}}\mathcal{L} \in \mathbb{R}^P`` em ``\mathcal{O}(1)`` passagens até a precisão de máquina, sem erros de truncamento de diferenças finitas.
"""

# ╔═╡ 10000000-0000-4000-8000-000000000007
let
    Random.seed!(123)
    f_escalar(x) = 3x^2 + 2x - 5
    x_ponto = 4.0f0

    (df_ad,) = Zygote.gradient(f_escalar, x_ponto)
    y_val, op_pullback = Zygote.pullback(f_escalar, x_ponto)
    (df_pb,) = op_pullback(1.0f0)

    N = 100
    X_sint = randn(Float32, 2, N)
    W_real, b_real = Float32[3.5 -1.8], Float32[0.7]
    y_sint = W_real * X_sint .+ b_real .+ 0.1f0 .* randn(Float32, 1, N)

    W_ini, b_ini = randn(Float32, 1, 2), zeros(Float32, 1)
    perda_mse, (grad_W, grad_b) = Zygote.withgradient(W_ini, b_ini) do W, b
        mean(((W * X_sint .+ b) .- y_sint) .^ 2)
    end

    grad_W_exato = (2.0f0 / N) .* (((W_ini * X_sint .+ b_ini) .- y_sint) * transpose(X_sint))
    erro_max = maximum(abs.(grad_W .- grad_W_exato))

    md"""
    - **Escalar em ``x=4``:** ``f(4)=`` `$(y_val)`, `Zygote.gradient` = `$(df_ad)`, `pullback(1.0f0)` = `$(df_pb)` *(exato: `26.0`)*
    - **Matricial (MSE = `$(round(perda_mse, digits=4))`):** ``\nabla_{\mathbf{W}}\mathcal{L} =`` `$(round.(grad_W, digits=5))` | ``\nabla_{\mathbf{b}}\mathcal{L} =`` `$(round.(grad_b, digits=5))`
    - **Erro máximo vs gradiente analítico:** **`$(erro_max)`**
    """
end

# ╔═╡ 10000000-0000-4000-8000-000000000008
md"""
---
## Nível 2.5: Funções de Ativação e o Neurônio Logístico

Sem ativações não-lineares, a composição ``\mathbf{W}_2(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2 = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2\mathbf{b}_1+\mathbf{b}_2)`` colapsa em uma única transformação afim. Abaixo comparamos quatro ativações e treinamos um único neurônio linear com entropia cruzada binária (equivalente à **Regressão Logística**).
"""

# ╔═╡ 10000000-0000-4000-8000-000000000009
let
    z_grid = collect(-4.5f0:0.05f0:4.5f0)
    ativacoes = [("Sigmoide", sigmoid, :royalblue), ("tanh", tanh, :darkorange), ("ReLU", relu, :crimson), ("GELU", gelu, :seagreen)]

    p_fn = plot(title="Ativações g(z)", xlabel="z", ylabel="g(z)", ylims=(-1.3, 3.0), legend=:topleft)
    p_df = plot(title="Derivadas g'(z) (Zygote)", xlabel="z", ylabel="g'(z)", ylims=(-0.3, 1.3), legend=:topleft)
    for (rot, fn, cor) in ativacoes
        plot!(p_fn, z_grid, fn.(z_grid), lw=2.5, color=cor, label=rot)
        plot!(p_df, z_grid, [first(Zygote.gradient(fn, z)) for z in z_grid], lw=2.5, color=cor, label=rot)
    end
    plot(p_fn, p_df, layout=(1, 2), size=(860, 310))
end

# ╔═╡ 10000000-0000-4000-8000-000000000010
let
    Random.seed!(314)
    n_c = 120
    X0 = randn(Float32, 2, n_c) .* 0.65f0 .- Float32[1.0; 0.8]
    X1 = randn(Float32, 2, n_c) .* 0.65f0 .+ Float32[1.0; 0.8]
    X_bin = hcat(X0, X1)
    y_bin = hcat(zeros(Float32, 1, n_c), ones(Float32, 1, n_c))

    modelo_log = Dense(2 => 1)
    est_log = Flux.setup(Optimisers.Adam(0.05f0), modelo_log)
    for _ in 1:150
        _, (g_log,) = Flux.withgradient(m -> Flux.logitbinarycrossentropy(m(X_bin), y_bin), modelo_log)
        Flux.update!(est_log, modelo_log, g_log)
    end

    acc = mean((sigmoid.(modelo_log(X_bin)) .>= 0.5f0) .== (y_bin .== 1.0f0)) * 100.0
    eixo = range(-3.0f0, 3.0f0, length=90)
    malha = reduce(hcat, [[x1, x2] for x2 in eixo, x1 in eixo])
    mapa = reshape(sigmoid.(modelo_log(malha)), length(eixo), length(eixo))

    p = contourf(eixo, eixo, mapa, levels=15, color=:RdBu_4, alpha=0.75, title="Regressão Logística (Acc: $(round(acc, digits=1))%)", xlabel="x₁", ylabel="x₂")
    contour!(p, eixo, eixo, mapa, levels=[0.5], lw=3, color=:black, colorbar=false)
    scatter!(p, X0[begin, :], X0[end, :], color=:crimson, label="Classe 0", ms=4)
    scatter!(p, X1[begin, :], X1[end, :], color=:royalblue, label="Classe 1", ms=4)
    plot(p, size=(600, 350))
end

# ╔═╡ 10000000-0000-4000-8000-000000000011
md"""
---
## Nível 3: Aproximação Universal e Regressão Não-Linear Reativa

Pelo Teorema da Aproximação Universal, combinações ``\hat{f}(x) = \sum_{m=1}^M v_m g(w_m x + b_m) + c`` aproximam qualquer função contínua em domínios compactos. Ajuste os controles abaixo:

- **Ativação:** $(@bind ativ_escolhida PlutoUI.Select(["tanh" => "tanh(z)", "gelu" => "GELU(z)", "relu" => "ReLU(z)"]; default="tanh"))
- **Largura oculta (``M``):** $(@bind n_ocultos_reg PlutoUI.Slider(4:4:96, default=32, show_value=true))
- **Épocas:** $(@bind epocas_reg PlutoUI.Slider(50:50:800, default=350, show_value=true))
- **``\log_{10}(\eta)``:** $(@bind log_lr_reg PlutoUI.Slider(-3.0:0.25:-1.5, default=-2.0, show_value=true))
"""

# ╔═╡ 10000000-0000-4000-8000-000000000012
let
    Random.seed!(2026)
    fn_ativ = ativ_escolhida == "relu" ? relu : (ativ_escolhida == "gelu" ? gelu : tanh)
    n_pts = 280
    x_reg = reshape(collect(range(-Float32(π), Float32(π), length=n_pts)), 1, n_pts)
    y_limpo = sin.(x_reg) .+ 0.35f0 .* cos.(3.0f0 .* x_reg)
    y_reg = y_limpo .+ 0.12f0 .* randn(Float32, 1, n_pts)

    modelo_reg = Chain(Dense(1 => n_ocultos_reg, fn_ativ), Dense(n_ocultos_reg => n_ocultos_reg, fn_ativ), Dense(n_ocultos_reg => 1))
    lr_ef = Float32(10.0 ^ log_lr_reg)
    est_reg = Flux.setup(Optimisers.Adam(lr_ef), modelo_reg)
    hist = Vector{Float32}(undef, epocas_reg)

    for ep in 1:epocas_reg
        perda, (g_reg,) = Flux.withgradient(m -> Flux.mse(m(x_reg), y_reg), modelo_reg)
        Flux.update!(est_reg, modelo_reg, g_reg)
        hist[ep] = perda
    end

    p1 = scatter(vec(x_reg), vec(y_reg), alpha=0.35, color=:steelblue, label="Dados", title="Ajuste (MSE=$(round(last(hist), digits=4)))")
    plot!(p1, vec(x_reg), vec(y_limpo), lw=2, ls=:dash, color=:black, label="Verdadeira")
    plot!(p1, vec(x_reg), vec(modelo_reg(x_reg)), lw=3, color=:crimson, label="Rede Flux")
    p2 = plot(hist, yscale=:log10, lw=2, color=:darkorange, title="Convergência", xlabel="Época", ylabel="MSE", legend=false)
    plot(p1, p2, layout=(1, 2), size=(860, 330))
end

# ╔═╡ 10000000-0000-4000-8000-000000000013
md"""
---
## Nível 4: MLP Multiclasse, Truque *Log-Sum-Exp*, `BatchNorm`, `Dropout` e `AdamW`

Ao manter a camada de saída linear e usar `Flux.logitcrossentropy`, avaliamos ``-\log\text{softmax}(\mathbf{z})_k = -z_k + c + \log\sum_j e^{z_j - c}`` com ``c = \max_j z_j``, evitando *overflow* numérico em `Float32`.
"""

# ╔═╡ 10000000-0000-4000-8000-000000000014
let
    Random.seed!(77)
    n_pc, nc = 250, 3
    X_esp = Matrix{Float32}(undef, 2, n_pc * nc)
    y_esp = Vector{Int}(undef, n_pc * nc)
    idx = 1
    for c in 1:nc, i in 1:n_pc
        r = Float32(i) / Float32(n_pc)
        t = Float32(c - 1) * 4.2f0 + 4.2f0 * r + randn(Float32) * 0.18f0
        X_esp[begin, idx] = r * sin(t)
        X_esp[end, idx]   = r * cos(t)
        y_esp[idx] = c
        idx += 1
    end
    Y_oh = onehotbatch(y_esp, 1:3)
    (X_tr, Y_tr), (X_val, Y_val) = splitobs((X_esp, Y_oh); at=0.8, shuffle=true)
    loader_tr = DataLoader((X_tr, Y_tr), batchsize=32, shuffle=true)

    modelo_mlp = Chain(
        Dense(2 => 64), BatchNorm(64, relu), Dropout(0.1f0),
        Dense(64 => 64), BatchNorm(64, relu), Dropout(0.1f0),
        Dense(64 => 3)
    )
    est_mlp = Flux.setup(Optimisers.AdamW(0.008f0, (0.9f0, 0.999f0), 1.0f-4), modelo_mlp)

    hist_tr, hist_val = Float32[], Float32[]
    for _ in 1:80
        Flux.trainmode!(modelo_mlp)
        for (xb, yb) in loader_tr
            _, (g_mlp,) = Flux.withgradient(m -> Flux.logitcrossentropy(m(xb), yb), modelo_mlp)
            Flux.update!(est_mlp, modelo_mlp, g_mlp)
        end
        Flux.testmode!(modelo_mlp)
        push!(hist_tr, Flux.logitcrossentropy(modelo_mlp(X_tr), Y_tr))
        push!(hist_val, Flux.logitcrossentropy(modelo_mlp(X_val), Y_val))
    end

    Flux.testmode!(modelo_mlp)
    preds_v, reais_v = onecold(modelo_mlp(X_val)), onecold(Y_val)
    acc_v = mean(preds_v .== reais_v) * 100.0
    mat_conf = [sum((preds_v .== i) .& (reais_v .== j)) for i in 1:3, j in 1:3]

    grade = range(-1.2f0, 1.2f0, length=110)
    pts = reduce(hcat, [[x, y] for y in grade, x in grade])
    cls = reshape(onecold(modelo_mlp(pts)), length(grade), length(grade))

    pf = heatmap(grade, grade, cls, alpha=0.35, color=[:royalblue, :darkorange, :seagreen], colorbar=false, title="MLP (Acc Val: $(round(acc_v, digits=1))%)")
    scatter!(pf, X_esp[begin, :], X_esp[end, :], group=y_esp, palette=[:royalblue, :darkorange, :seagreen], ms=3, label=["C1" "C2" "C3"])
    pp = plot(hist_tr, lw=2, label="Treino", title="Entropia Cruzada", xlabel="Época")
    plot!(pp, hist_val, lw=2, ls=:dash, label="Validação")
    pc = heatmap(["R1", "R2", "R3"], ["P1", "P2", "P3"], mat_conf, color=:Blues, colorbar=false, yflip=true, title="Confusão (Val)")
    for i in 1:3, j in 1:3
        annotate!(pc, j, i, text("$(mat_conf[i, j])", 10, :black))
    end
    plot(pf, pp, pc, layout=(1, 3), size=(960, 300))
end

# ╔═╡ 10000000-0000-4000-8000-000000000015
md"""
---
## Nível 5: CNN com Bloco Residual (`Flux.@layer`), *ClipNorm* e *Cosine Annealing*

Em imagens 2D, o `Flux.jl` adota a convenção **`(Largura, Altura, Canais, Batch)`**. Conexões residuais ``\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}`` introduzem a identidade ``\mathbf{I}`` na Jacobiana ``\partial\mathbf{y}/\partial\mathbf{x} = \mathbf{I} + \partial\mathcal{F}/\partial\mathbf{x}``, preservando o sinal de erro até as camadas iniciais.
"""

# ╔═╡ 10000000-0000-4000-8000-000000000016
begin
    struct BlocoResidualLeve{C1, B1, C2, B2}
        conv1::C1
        bn1::B1
        conv2::C2
        bn2::B2
    end
    BlocoResidualLeve(c::Int) = BlocoResidualLeve(
        Conv((3, 3), c => c, pad=1, bias=false), BatchNorm(c),
        Conv((3, 3), c => c, pad=1, bias=false), BatchNorm(c)
    )
    Flux.@layer BlocoResidualLeve
    function (b::BlocoResidualLeve)(x)
        h = relu.(b.bn1(b.conv1(x)))
        return relu.(b.bn2(b.conv2(h)) .+ x)
    end
end

# ╔═╡ 10000000-0000-4000-8000-000000000017
let
    Random.seed!(2026)
    n_am = 360
    X_img = 0.18f0 .* randn(Float32, 16, 16, 1, n_am)
    y_img = Vector{Int}(undef, n_am)
    for i in 1:n_am
        c = mod1(i, 3)
        y_img[i] = c
        p = clamp(8 + rand(-3:3), 4, 13)
        if c == 1
            X_img[:, (p-1):(p+1), begin, i] .+= 1.0f0
        elseif c == 2
            X_img[(p-1):(p+1), :, begin, i] .+= 1.0f0
        else
            X_img[:, p, begin, i] .+= 1.0f0
            X_img[p, :, begin, i] .+= 1.0f0
        end
    end
    Y_img_oh = onehotbatch(y_img, 1:3)
    (X_itr, Y_itr), (X_ival, Y_ival) = splitobs((X_img, Y_img_oh); at=0.8, shuffle=true)
    loader_cnn = DataLoader((X_itr, Y_itr), batchsize=32, shuffle=true)

    modelo_cnn = Chain(
        Conv((3, 3), 1 => 8, pad=1, bias=false), BatchNorm(8, relu), MaxPool((2, 2)),
        BlocoResidualLeve(8),
        Conv((3, 3), 8 => 16, pad=1, bias=false), BatchNorm(16, relu), MaxPool((2, 2)),
        Flux.flatten, Dropout(0.2f0), Dense(4 * 4 * 16 => 32, relu), Dense(32 => 3)
    )

    est_cnn = Flux.setup(Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.AdamW(0.005f0)), modelo_cnn)
    melhor_perda, ep_melhor = Inf32, 1
    melhor_estado = deepcopy(Flux.state(modelo_cnn))
    epocas_cnn, lr_max, lr_min = 15, 0.005f0, 1.0f-4
    hist_lr, hist_p = Float32[], Float32[]

    for ep in 1:epocas_cnn
        Flux.trainmode!(modelo_cnn)
        lr_at = lr_min + 0.5f0 * (lr_max - lr_min) * (1.0f0 + cos(Float32(π) * (ep - 1) / epocas_cnn))
        Optimisers.adjust!(est_cnn, lr_at)
        push!(hist_lr, lr_at)

        soma, nb = 0.0f0, 0
        for (xb, yb) in loader_cnn
            pb, (g_cnn,) = Flux.withgradient(m -> Flux.logitcrossentropy(m(xb), yb), modelo_cnn)
            Flux.update!(est_cnn, modelo_cnn, g_cnn)
            soma += pb; nb += 1
        end
        p_med = soma / nb
        push!(hist_p, p_med)
        if p_med < melhor_perda
            melhor_perda, ep_melhor = p_med, ep
            melhor_estado = deepcopy(Flux.state(modelo_cnn))
        end
    end

    Flux.loadmodel!(modelo_cnn, melhor_estado)
    Flux.testmode!(modelo_cnn)
    acc_cnn = mean(onecold(modelo_cnn(X_ival)) .== onecold(Y_ival)) * 100.0

    i1, i2, i3 = findfirst(==(1), y_img), findfirst(==(2), y_img), findfirst(==(3), y_img)
    pe1 = heatmap(X_img[:, :, begin, i1]', color=:grays, colorbar=false, ticks=false, yflip=true, title="C1: Horiz.")
    pe2 = heatmap(X_img[:, :, begin, i2]', color=:grays, colorbar=false, ticks=false, yflip=true, title="C2: Vert.")
    pe3 = heatmap(X_img[:, :, begin, i3]', color=:grays, colorbar=false, ticks=false, yflip=true, title="C3: Cruz")
    pt = plot(hist_p, lw=2.5, color=:crimson, label="Treino", title="CNN Residual (Acc Val: \$(round(acc_cnn, digits=1))% | Melhor Época: \$(ep_melhor))")
    ps = plot(hist_lr, lw=2.5, color=:purple, label="η(t)", title="Cosine Annealing LR")
    plot(pe1, pe2, pe3, pt, ps, layout=@layout([grid(1, 3){0.35h}; grid(1, 2)]), size=(880, 480))
end

# ╔═╡ 10000000-0000-4000-8000-000000000018
md"""
---
## Nível 6: Autoencoder Não-Linear vs PCA (Aprendizado de Variedades)

Minimizando ``\frac{1}{N}\sum_i \|\mathbf{x}_i - g_{\boldsymbol{\theta}}(f_{\boldsymbol{\phi}}(\mathbf{x}_i))\|_2^2`` com gargalo ``\mathbb{R}^3 \to \mathbb{R}^1 \to \mathbb{R}^3``, um Autoencoder não-linear generaliza o subespaço reto da PCA para parametrizar variedades curvas imersas.
"""

# ╔═╡ 10000000-0000-4000-8000-000000000019
let
    Random.seed!(2026)
    N_man = 300
    t_real = collect(range(0.0f0, 2.2f0 * Float32(π), length=N_man))
    X_man = vcat(cos.(t_real)', sin.(t_real)', (0.35f0 .* t_real)') .+ 0.06f0 .* randn(Float32, 3, N_man)

    mu_X = mean(X_man, dims=2)
    X_cent = X_man .- mu_X
    u1 = svd(X_cent).U[:, begin:begin]
    X_pca = (u1 * (u1' * X_cent)) .+ mu_X
    mse_pca = Flux.mse(X_pca, X_man)

    cod = Chain(Dense(3 => 32, tanh), Dense(32 => 16, tanh), Dense(16 => 1))
    dec = Chain(Dense(1 => 16, tanh), Dense(16 => 32, tanh), Dense(32 => 3))
    ae  = Chain(cod, dec)
    est_ae = Flux.setup(Optimisers.Adam(0.01f0), ae)
    for _ in 1:600
        _, (g_ae,) = Flux.withgradient(m -> Flux.mse(m(X_cent), X_cent), ae)
        Flux.update!(est_ae, ae, g_ae)
    end

    z_lat = vec(cod(X_cent))
    X_ae = ae(X_cent) .+ mu_X
    mse_ae = Flux.mse(X_ae, X_man)
    X_ae_ord = X_ae[:, sortperm(z_lat)]

    eixo_x(M) = view(M, 1, :)
    eixo_y(M) = view(M, 2, :)
    eixo_z(M) = view(M, 3, :)

    p3d = scatter3d(eixo_x(X_man), eixo_y(X_man), eixo_z(X_man), zcolor=t_real, ms=3, alpha=0.45, colorbar=false, title="MSE: PCA=\$(round(mse_pca, digits=3)) vs AE=\$(round(mse_ae, digits=4))", label="Dados")
    plot3d!(p3d, eixo_x(X_pca), eixo_y(X_pca), eixo_z(X_pca), lw=3, ls=:dash, color=:black, label="PCA (1D)")
    plot3d!(p3d, eixo_x(X_ae_ord), eixo_y(X_ae_ord), eixo_z(X_ae_ord), lw=4, color=:crimson, label="Autoencoder (1D)")
    plat = scatter(t_real, z_lat, zcolor=t_real, ms=4, colorbar=false, legend=false, title="Coordenada Latente z", xlabel="t verdadeiro", ylabel="z ∈ ℝ¹")
    plot(p3d, plat, layout=(1, 2), size=(880, 360))
end

# ╔═╡ 10000000-0000-4000-8000-000000000020
md"""
---
## Nível 7: Redes Neurais Recorrentes (`RNN` vs `LSTM`) e Séries Temporais

Enquanto a `RNN` de Elman aplica ``\mathbf{h}_t = \tanh(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h)`` — sofrendo desvanecimento exponencial do gradiente no BPTT ``\prod_t \operatorname{diag}(1 - \mathbf{h}_t^{\odot 2})\mathbf{W}_{hh}^\top`` —, a `LSTM` mantém um estado de célula aditivo ``\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t`` modulado por portões sigmoidais. No `Flux.jl` atual, ambas operam de modo *stateless* sobre tensores 3D **`(Atributos, Janela, Batch)`**.
"""

# ╔═╡ 10000000-0000-4000-8000-000000000021
md"""
**🎛️ Painel Interativo de Séries Temporais (`RNN` vs `LSTM`):**
- **Janela Histórica (``L`` passos):** $(@bind janela_seq PlutoUI.Slider(12:6:48, default=30, show_value=true))
- **Estado Oculto (``H``):** $(@bind n_ocultos_seq PlutoUI.Slider(8:8:48, default=24, show_value=true))
- **Horizonte Autoregressivo (``K`` passos):** $(@bind horiz_autoregr PlutoUI.Slider(20:10:80, default=50, show_value=true))
"""

# ╔═╡ 10000000-0000-4000-8000-000000000022
let
    Random.seed!(2026)

    T_total = 420
    tempo = collect(range(0.0f0, 42.0f0, length=T_total))
    serie_limpa = (1.0f0 .+ 0.45f0 .* cos.(0.18f0 .* tempo)) .* sin.(0.85f0 .* tempo) .+ 0.30f0 .* sin.(2.3f0 .* tempo)
    serie_obs = serie_limpa .+ 0.06f0 .* randn(Float32, T_total)

    function criar_janelas_3d(s::Vector{Float32}, L::Int)
        N = length(s) - L
        X = Array{Float32, 3}(undef, 1, L, N)
        Y = Matrix{Float32}(undef, 1, N)
        for i in 1:N
            X[begin, :, i] = s[i:(i + L - 1)]
            Y[begin, i]    = s[i + L]
        end
        return X, Y
    end

    X_seq, Y_seq = criar_janelas_3d(serie_obs, janela_seq)
    N_janelas = size(X_seq, 3)

    n_treino = floor(Int, 0.72 * N_janelas)
    X_tr_seq, Y_tr_seq = X_seq[:, :, 1:n_treino], Y_seq[:, 1:n_treino]
    X_te_seq, Y_te_seq = X_seq[:, :, (n_treino + 1):end], Y_seq[:, (n_treino + 1):end]

    loader_seq = DataLoader((X_tr_seq, Y_tr_seq), batchsize=32, shuffle=true)

    modelo_rnn  = Chain(Flux.RNN(1 => n_ocultos_seq, tanh), H -> H[:, end, :], Dense(n_ocultos_seq => 1))
    modelo_lstm = Chain(Flux.LSTM(1 => n_ocultos_seq),      H -> H[:, end, :], Dense(n_ocultos_seq => 1))

    regra_seq = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.Adam(0.008f0))
    est_rnn  = Flux.setup(regra_seq, modelo_rnn)
    est_lstm = Flux.setup(regra_seq, modelo_lstm)

    epocas_seq = 65
    hist_rnn, hist_lstm = Vector{Float32}(undef, epocas_seq), Vector{Float32}(undef, epocas_seq)

    for ep in 1:epocas_seq
        sr, sl, nb = 0.0f0, 0.0f0, 0
        for (xb, yb) in loader_seq
            pr, (gr,) = Flux.withgradient(m -> Flux.mse(m(xb), yb), modelo_rnn)
            Flux.update!(est_rnn, modelo_rnn, gr)
            pl, (gl,) = Flux.withgradient(m -> Flux.mse(m(xb), yb), modelo_lstm)
            Flux.update!(est_lstm, modelo_lstm, gl)
            sr += pr; sl += pl; nb += 1
        end
        hist_rnn[ep], hist_lstm[ep] = sr / nb, sl / nb
    end

    pred_1p_rnn  = vec(modelo_rnn(X_te_seq))
    pred_1p_lstm = vec(modelo_lstm(X_te_seq))
    mse_1p_rnn   = Flux.mse(pred_1p_rnn, vec(Y_te_seq))
    mse_1p_lstm  = Flux.mse(pred_1p_lstm, vec(Y_te_seq))

    function prever_autoregressivo(modelo, janela_inicial::Array{Float32, 3}, passos::Int)
        janela = copy(janela_inicial)
        preds = Vector{Float32}(undef, passos)
        for k in 1:passos
            y_prox = first(modelo(janela))
            preds[k] = y_prox
            janela = cat(janela[:, 2:end, :], fill(y_prox, 1, 1, 1); dims=2)
        end
        return preds
    end

    K_efetivo = min(horiz_autoregr, size(X_te_seq, 3))
    janela_ini = X_te_seq[:, :, begin:begin]
    rollout_rnn  = prever_autoregressivo(modelo_rnn,  janela_ini, K_efetivo)
    rollout_lstm = prever_autoregressivo(modelo_lstm, janela_ini, K_efetivo)

    idx_teste = (n_treino + janela_seq + 1):T_total
    idx_roll  = idx_teste[1:K_efetivo]
    t0_roll   = first(idx_roll)
    mse_roll_rnn  = Flux.mse(rollout_rnn,  serie_limpa[idx_roll])
    mse_roll_lstm = Flux.mse(rollout_lstm, serie_limpa[idx_roll])

    p1 = plot(tempo[idx_teste], serie_limpa[idx_teste], lw=2.5, ls=:dash, color=:black, label="Real",
              title="1-Passo à Frente (MSE: RNN=\$(round(mse_1p_rnn, digits=4)) | LSTM=\$(round(mse_1p_lstm, digits=4)))")
    plot!(p1, tempo[idx_teste], pred_1p_rnn,  lw=1.8, color=:darkorange, label="RNN")
    plot!(p1, tempo[idx_teste], pred_1p_lstm, lw=2.0, color=:royalblue,  label="LSTM")

    idx_ctx = max(1, t0_roll - 35):(t0_roll - 1)
    p2 = plot(tempo[idx_ctx], serie_obs[idx_ctx], lw=2, color=:gray50, label="Histórico",
              title="Rollout (\$(K_efetivo) passos | MSE: RNN=\$(round(mse_roll_rnn, digits=3)) vs LSTM=$(round(mse_roll_lstm, digits=3)))")
    vline!(p2, [tempo[t0_roll]], ls=:dot, lw=1.5, color=:black, label="Início")
    plot!(p2, tempo[idx_roll], serie_limpa[idx_roll], lw=2.5, ls=:dash, color=:black, label="Futuro real")
    plot!(p2, tempo[idx_roll], rollout_rnn,  lw=2.2, color=:darkorange, label="Rollout RNN")
    plot!(p2, tempo[idx_roll], rollout_lstm, lw=2.5, color=:crimson,    label="Rollout LSTM")

    p3 = plot(hist_rnn, yscale=:log10, lw=2, color=:darkorange, label="RNN", title="Convergência BPTT", xlabel="Época")
    plot!(p3, hist_lstm, yscale=:log10, lw=2.2, color=:royalblue, label="LSTM")

    plot(p1, p2, p3, layout=@layout([a; b{0.58w} c{0.42w}]), size=(920, 520))
end

# ╔═╡ 10000000-0000-4000-8000-000000000023
md"""
---
## Nível 8: Mecanismos de Atenção, Codificação Posicional e Transformers

Nas arquiteturas recorrentes (Nível 7), toda informação histórica atravessa sequencialmente o vetor fixo ``\mathbf{h}_t``, exigindo ``\mathcal{O}(L)`` passos entre tokens distantes. O mecanismo de **Autoatenção (*Self-Attention*)** (Vaswani et al., 2017) conecta quaisquer duas posições em **``\mathcal{O}(1)`` saltos**, projetando ``\mathbf{X} \in \mathbb{R}^{d_{\text{model}} \times L}`` em **Consultas** (``\mathbf{Q}``), **Chaves** (``\mathbf{K}``) e **Valores** (``\mathbf{V}``).

### Atenção pelo Produto Escalar Escalado em Layout *Column-Major*

``\mathbf{S} = \frac{\mathbf{K}^\top \mathbf{Q}}{\sqrt{d_k}} + \mathbf{M}_{\text{causal}}, \qquad \boldsymbol{\alpha} = \operatorname{softmax}_{\text{col}}(\mathbf{S}), \qquad \mathbf{Z} = \mathbf{V}\boldsymbol{\alpha}``

- **Escala ``1/\sqrt{d_k}``:** Se ``\mathbf{q}_j`` e ``\mathbf{k}_i`` têm componentes de variância unitária, ``\operatorname{Var}(\mathbf{k}_i^\top \mathbf{q}_j) = d_k``; dividir por ``\sqrt{d_k}`` preserva variância unitária e evita saturação da `softmax`.
- **Dualidade da Máscara Causal em Julia:** Como `softmax(..., dims=1)` normaliza **colunas** (onde a coluna ``j`` é a consulta no instante ``j`` e a linha ``i`` é a chave no instante ``i``), impedir acesso ao futuro ``i > j`` corresponde a mascarar o **triângulo estritamente inferior** (``M_{i,j} = -\infty`` para ``i > j``).
- **Codificação Posicional:** Ondas determinísticas ``\operatorname{PE}_{(2m, t)} = \sin((t-1)/10000^{2m/d})`` e ``\operatorname{PE}_{(2m+1, t)} = \cos((t-1)/10000^{2m/d})`` quebram a simetria de permutação da atenção.
"""

# ╔═╡ 10000000-0000-4000-8000-000000000024
begin
    function codificacao_posicional_senoidal(d_model::Int, tam_max::Int)
        PE = zeros(Float32, d_model, tam_max)
        for pos in 1:tam_max, i in 1:2:(d_model - 1)
            div_termo = Float32(10000.0 ^ ((i - 1) / d_model))
            angulo = Float32(pos - 1) / div_termo
            PE[i, pos]     = sin(angulo)
            PE[i + 1, pos] = cos(angulo)
        end
        return PE
    end

    struct AtencaoMultiCabeca{Q, K, V, O}
        wq::Q
        wk::K
        wv::V
        wo::O
        n_cabecas::Int
    end

    function AtencaoMultiCabeca(d_model::Int, n_cabecas::Int)
        @assert d_model % n_cabecas == 0 "d_model deve ser múltiplo de n_cabecas"
        return AtencaoMultiCabeca(
            Dense(d_model => d_model, bias=false),
            Dense(d_model => d_model, bias=false),
            Dense(d_model => d_model, bias=false),
            Dense(d_model => d_model, bias=false),
            n_cabecas
        )
    end

    Flux.@layer AtencaoMultiCabeca

    function (mha::AtencaoMultiCabeca)(X::AbstractArray{Float32, 3}; causal::Bool=true, retornar_pesos::Bool=false)
        d_model, L, B = size(X)
        H = mha.n_cabecas
        d_k = div(d_model, H)
        escala = Float32(1.0 / sqrt(d_k))

        Q3 = reshape(permutedims(reshape(mha.wq(X), d_k, H, L, B), (1, 3, 2, 4)), d_k, L, H * B)
        K3 = reshape(permutedims(reshape(mha.wk(X), d_k, H, L, B), (1, 3, 2, 4)), d_k, L, H * B)
        V3 = reshape(permutedims(reshape(mha.wv(X), d_k, H, L, B), (1, 3, 2, 4)), d_k, L, H * B)

        escores = Flux.NNlib.batched_mul(permutedims(K3, (2, 1, 3)), Q3) .* escala

        if causal
            mascara = Zygote.@ignore [i > j ? -1.0f9 : 0.0f0 for i in 1:L, j in 1:L]
            escores = escores .+ mascara
        end

        alfa = softmax(escores; dims=1)
        ctx3 = Flux.NNlib.batched_mul(V3, alfa)

        ctx_cat = reshape(permutedims(reshape(ctx3, d_k, L, H, B), (1, 3, 2, 4)), d_model, L, B)
        saida = mha.wo(ctx_cat)

        return retornar_pesos ? (saida, reshape(alfa, L, L, H, B)) : saida
    end

    struct BlocoTransformerCausal{N1, A, N2, F}
        ln1::N1
        atencao::A
        ln2::N2
        ffn::F
    end

    function BlocoTransformerCausal(d_model::Int, n_cabecas::Int; d_ff::Int = 2 * d_model, p_drop::Float32 = 0.05f0)
        return BlocoTransformerCausal(
            LayerNorm(d_model),
            AtencaoMultiCabeca(d_model, n_cabecas),
            LayerNorm(d_model),
            Chain(Dense(d_model => d_ff, gelu), Dropout(p_drop), Dense(d_ff => d_model))
        )
    end

    Flux.@layer BlocoTransformerCausal

    function (bloco::BlocoTransformerCausal)(X::AbstractArray{Float32, 3})
        H = X .+ bloco.atencao(bloco.ln1(X); causal=true)
        return H .+ bloco.ffn(bloco.ln2(H))
    end
end

# ╔═╡ 10000000-0000-4000-8000-000000000025
let
    Random.seed!(2026)

    # Tarefa de Memória Associativa (Induction Heads): [k₁, v₁, k₂, v₂, k₃, v₃, '?', k_q] -> prever v_q!
    function gerar_associacoes(n_amostras::Int)
        X = Matrix{Int}(undef, 8, n_amostras)
        Y = Vector{Int}(undef, n_amostras)
        for n in 1:n_amostras
            chaves  = randperm(4)[1:3]
            valores = rand(5:8, 3)
            for par in 1:3
                X[2 * par - 1, n] = chaves[par]
                X[2 * par, n]     = valores[par]
            end
            X[7, n] = 9
            escolhida = rand(1:3)
            X[8, n] = chaves[escolhida]
            Y[n]    = valores[escolhida] - 4
        end
        return X, Y
    end

    X_assoc, y_assoc = gerar_associacoes(480)
    Y_assoc_oh = onehotbatch(y_assoc, 1:4)
    (X_atr, Y_atr), (X_aval, Y_aval) = splitobs((X_assoc, Y_assoc_oh); at=0.8, shuffle=true)
    loader_assoc = DataLoader((X_atr, Y_atr), batchsize=32, shuffle=true)

    d_mod = 32
    PE_fixo = codificacao_posicional_senoidal(d_mod, 8)
    modelo_assoc = (
        emb  = Embedding(9 => d_mod),
        b1   = BlocoTransformerCausal(d_mod, 4; p_drop=0.0f0),
        b2   = BlocoTransformerCausal(d_mod, 4; p_drop=0.0f0),
        head = Dense(d_mod => 4)
    )

    est_assoc = Flux.setup(Optimisers.Adam(0.008f0), modelo_assoc)

    for _ in 1:75
        for (xb, yb) in loader_assoc
            _, (g_assoc,) = Flux.withgradient(modelo_assoc) do m
                h2 = m.b2(m.b1(m.emb(xb) .+ PE_fixo))
                Flux.logitcrossentropy(m.head(h2[:, end, :]), yb)
            end
            Flux.update!(est_assoc, modelo_assoc, g_assoc)
        end
    end

    h_val = modelo_assoc.b2(modelo_assoc.b1(modelo_assoc.emb(X_aval) .+ PE_fixo))
    acc_assoc = mean(onecold(modelo_assoc.head(h_val[:, end, :])) .== onecold(Y_aval)) * 100.0

    amostra_ex = X_aval[:, begin:begin]
    h1_ex = modelo_assoc.b1(modelo_assoc.emb(amostra_ex) .+ PE_fixo)
    _, pesos_ex = modelo_assoc.b2.atencao(modelo_assoc.b2.ln1(h1_ex); causal=true, retornar_pesos=true)
    mapa_atencao = dropdims(mean(pesos_ex[:, :, :, begin], dims=3), dims=3)'

    nomes_simb = Dict(1=>"K1", 2=>"K2", 3=>"K3", 4=>"K4", 5=>"V1", 6=>"V2", 7=>"V3", 8=>"V4", 9=>"?")
    rotulos_seq = ["$(t):$(nomes_simb[amostra_ex[t, begin]])" for t in 1:8]

    p_pe = heatmap(
        codificacao_posicional_senoidal(32, 32), color=:RdBu,
        title="Codificação Posicional Senoidal (32×32)",
        xlabel="Posição t", ylabel="Canal do Embedding"
    )
    p_att = heatmap(
        rotulos_seq, rotulos_seq, mapa_atencao, color=:viridis, yflip=true,
        title="Induction Head Camada 2 (Acc Val: $(round(acc_assoc, digits=1))%)",
        xlabel="Posição Atendida (Passado/Chave)", ylabel="Posição da Consulta"
    )
    plot(p_pe, p_att, layout=(1, 2), size=(900, 360))
end

# ╔═╡ 10000000-0000-4000-8000-000000000026
md"""
---
## Nível 9: Processamento de Linguagem Natural e Construção de uma SLM Machadiana

Um **Modelo de Linguagem Pequeno (*Small Language Model* — SLM)** do tipo GPT (*Decoder-Only Transformer*) fatora a probabilidade conjunta de uma sequência de tokens ``(w_1, w_2, \dots, w_T)`` via regra da cadeia:

``P_{\boldsymbol{\theta}}(w_1, \dots, w_T) = \prod_{t=1}^{T} P_{\boldsymbol{\theta}}(w_t \mid w_1, \dots, w_{t-1}), \qquad \operatorname{PPL}(\boldsymbol{\theta}) = \exp(\mathcal{L}_{\text{CE}})``

### Passo a Passo Implementado Abaixo:
1. **Tokenização Léxica Reversível + Tokens de Controle:** Isolamos sinais de pontuação e adicionamos `<pad>`, `<unk>`, `<pergunta>`, `<machado>` e `<fim>` sobre trechos em domínio público de **Machado de Assis** (*Dom Casmurro*, *Memórias Póstumas de Brás Cubas*, *Quincas Borba* e *O Alienista*).
2. **Fase 1 — Pré-Treinamento Autoregressivo:** A SLM absorve vocabulário, sintaxe e ritmo machadiano prevendo ``w_{t+1}`` em janelas contínuas da prosa literária.
3. **Fase 2 — *Instruction SFT* com Máscara de Perda:** Ajustamos o modelo em diálogos `<pergunta> ... <machado> ... <fim>` aplicando **Loss Masking** (o gradiente incide apenas sobre os tokens de resposta após `<machado>`).
4. **Decodificação com Temperatura (``\tau``) e *Top-k*:** Reescalamos ``\mathbf{p} = \operatorname{softmax}(\mathbf{z}/\tau)`` entre os ``k`` tokens mais prováveis em tempo real.
"""

# ╔═╡ 10000000-0000-4000-8000-000000000027
begin
    struct MiniGPT{E, P, B, N, C}
        tok_emb::E
        pos_emb::P
        blocos::B
        ln_final::N
        cabeca_saida::C
    end

    function MiniGPT(tam_vocab::Int, d_model::Int, n_cabecas::Int, n_camadas::Int, tam_ctx::Int)
        blocos = Chain([BlocoTransformerCausal(d_model, n_cabecas; p_drop=0.05f0) for _ in 1:n_camadas]...)
        return MiniGPT(
            Embedding(tam_vocab => d_model),
            Embedding(tam_ctx => d_model),
            blocos,
            LayerNorm(d_model),
            Dense(d_model => tam_vocab)
        )
    end

    Flux.@layer MiniGPT

    function (m::MiniGPT)(idx::AbstractMatrix{<:Integer})
        L = size(idx, 1)
        posicoes = Zygote.@ignore collect(1:L)
        X = m.tok_emb(idx) .+ m.pos_emb(posicoes)
        return m.cabeca_saida(m.ln_final(m.blocos(X)))
    end

    function tokenizar_machado(texto::AbstractString)
        limpo = replace(lowercase(strip(texto)), r"([,.;:?!—])" => s" \1 ")
        return String.(split(limpo))
    end

    function detokenizar_machado(tokens::Vector{String})
        nomes_proprios = Set(["capitu", "bentinho", "escobar", "brás", "cubas", "quincas", "borba", "rubião", "humanitas", "humanitismo", "simão", "bacamarte", "itaguaí", "marcela", "matacavalos"])
        partes = String[]
        cap_proxima = true
        for tok in tokens
            tok in ("<pad>", "<unk>", "<pergunta>", "<machado>", "<fim>") && continue
            if tok in (",", ".", ";", ":", "?", "!")
                isempty(partes) ? push!(partes, tok) : (partes[end] *= tok)
                cap_proxima = tok in (".", "?", "!")
            elseif tok == "—"
                push!(partes, "—")
            else
                palavra = (cap_proxima || (tok in nomes_proprios)) ? uppercasefirst(tok) : tok
                push!(partes, palavra)
                cap_proxima = false
            end
        end
        return join(partes, " ")
    end

    function construir_e_treinar_slm()
        Random.seed!(1839)

        prosa_machado = [
            "Eu não sou propriamente um autor comum, mas um narrador póstumo, para quem a memória foi outro berço.",
            "Dedico estas memórias como saudosa lembrança das minhas antigas reflexões e ironias humanas.",
            "Marcela estimou-me durante quinze meses e onze contos de réis; nada menos.",
            "O emplasto Brás Cubas era a minha ideia fixa para aliviar a melancolia da humanidade e alcançar a glória.",
            "Deus te livre, caro leitor, de uma ideia fixa; antes um argueiro, antes uma trave no olhar.",
            "Capitu tinha olhos de ressaca, olhos de cigana oblíqua e dissimulada.",
            "Traziam não sei que fluido misterioso e enérgico, uma força que arrastava para dentro, como a vaga que se retira da praia.",
            "O meu fim evidente era atar as duas pontas da vida, e restaurar na velhice a adolescência; pois, senhor, não consegui recompor o que foi nem o que fui.",
            "Ao vencedor, as batatas! Supõe tu um campo de batatas e duas tribos diante da escassez.",
            "Segundo a doutrina de Quincas Borba, a disputa conserva o vigor, pois Humanitas é o princípio único e universal.",
            "Cada homem é uma redução de Humanitas, e a bolha na água logo retorna ao grande oceano.",
            "A ciência, disse Simão Bacamarte, é o meu emprego único; Itaguaí é o meu universo.",
            "A loucura, objeto dos meus estudos, era até agora uma ilha perdida no oceano da razão; começo a suspeitar que é um continente.",
            "Simão Bacamarte recolheu-se à Casa Verde, convencido de que o equilíbrio perfeito da razão era uma rara exceção.",
            "O tempo é um tecido invisível em que se pode bordar tudo: uma flor, um pássaro, uma dama, um castelo, uma saudade.",
            "A vaidade é um estímulo amável e irônico que consola a nossa antiga mediocridade.",
            "Passamos as horas com silenciosa polidez, enquanto os séculos dobram a esquina da memória."
        ]

        pares_sft = [
            ("como eram os olhos de capitu?", "Capitu trazia olhos de ressaca, olhos de cigana oblíqua e dissimulada, com uma força misteriosa que arrastava para dentro como a vaga da praia."),
            ("o que pensa sobre o amor e capitu?", "O amor é um fluido misterioso e oblíquo; Marcela durou quinze meses e onze contos de réis, mas os olhos de Capitu arrastavam como a ressaca."),
            ("qual é a filosofia do humanitismo e das batatas?", "Ao vencedor, as batatas! Segundo Quincas Borba, cada homem é uma redução de Humanitas diante do vasto campo da existência."),
            ("o que buscava brás cubas em suas memórias?", "Como narrador de memórias póstumas, busquei no emplasto a glória humana, notando com fina ironia como a vaidade consola a nossa melancolia."),
            ("qual é a fronteira entre a loucura e a razão?", "Dizia Simão Bacamarte em Itaguaí que a loucura não é uma ilha perdida no oceano da razão, mas um vasto continente onde todos habitamos."),
            ("como o tempo trata os homens e a memória?", "O tempo é um tecido invisível em que se pode bordar uma flor, um castelo ou uma saudade, sempre com a polidez silenciosa dos séculos."),
            ("o que move a vaidade e a glória humana?", "A vaidade é um estímulo amável; até o meu emplasto Brás Cubas não passava de uma ideia fixa para conquistar renome entre os homens."),
            ("é possível recuperar o passado na velhice?", "Quis atar as duas pontas da vida e restaurar na velhice a adolescência; pois, caro leitor, não consegui recompor o que foi nem o que fui."),
            ("quem foi simão bacamarte na casa verde?", "Simão Bacamarte fez da ciência o seu universo em Itaguaí e estudou a Casa Verde até concluir que o perfeito equilíbrio era uma exceção."),
            ("qual conselho deixa ao leitor sobre ideias fixas?", "Deus te livre, caro leitor, de uma ideia fixa; a vida é oblíqua e costuma sorrir das nossas graves teorias com silenciosa polidez.")
        ]

        especiais = ["<pad>", "<unk>", "<pergunta>", "<machado>", "<fim>"]
        contagem = Dict{String, Int}()
        for t in prosa_machado, w in tokenizar_machado(t)
            contagem[w] = get(contagem, w, 0) + 1
        end
        for (p, r) in pares_sft, w in vcat(tokenizar_machado(p), tokenizar_machado(r))
            contagem[w] = get(contagem, w, 0) + 1
        end

        palavras_ord = sort(collect(keys(contagem)); by = w -> -contagem[w])
        id2tok = vcat(especiais, palavras_ord)
        tok2id = Dict(w => i for (i, w) in enumerate(id2tok))
        freq_zipf = [contagem[w] for w in palavras_ord]

        V = length(id2tok)
        tam_ctx = 28
        codificar(toks) = [get(tok2id, w, tok2id["<unk>"]) for w in toks]

        fluxo_pre = Int[]
        for frase in prosa_machado
            append!(fluxo_pre, codificar(tokenizar_machado(frase)))
        end
        starts = collect(1:3:(length(fluxo_pre) - tam_ctx - 1))
        X_pre = Matrix{Int}(undef, tam_ctx, length(starts))
        Y_pre = Matrix{Int}(undef, tam_ctx, length(starts))
        M_pre = ones(Float32, tam_ctx, length(starts))
        for (j, s) in enumerate(starts)
            X_pre[:, j] = fluxo_pre[s:(s + tam_ctx - 1)]
            Y_pre[:, j] = fluxo_pre[(s + 1):(s + tam_ctx)]
        end

        id_pad = tok2id["<pad>"]
        X_sft = fill(id_pad, tam_ctx, length(pares_sft) * 2)
        Y_sft = fill(id_pad, tam_ctx, length(pares_sft) * 2)
        M_sft = zeros(Float32, tam_ctx, length(pares_sft) * 2)

        col = 1
        for (p_txt, r_txt) in pares_sft
            toks_p = vcat(["<pergunta>"], tokenizar_machado(p_txt), ["<machado>"])
            toks_r = vcat(tokenizar_machado(r_txt), ["<fim>"])
            seq = codificar(vcat(toks_p, toks_r))
            idx_inicio_resp = length(toks_p)

            for desloc in (0, max(0, length(seq) - tam_ctx - 1))
                trecho = seq[(1 + desloc):min(end, desloc + tam_ctx + 1)]
                comp = length(trecho) - 1
                X_sft[1:comp, col] = trecho[1:comp]
                Y_sft[1:comp, col] = trecho[2:(comp + 1)]
                for t in 1:comp
                    if (t + desloc) >= idx_inicio_resp
                        M_sft[t, col] = 1.0f0
                    end
                end
                col += 1
            end
        end

        modelo = MiniGPT(V, 64, 4, 2, tam_ctx)
        regra = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.AdamW(0.006f0, (0.9f0, 0.99f0), 1.0f-4))
        estado_otim = Flux.setup(regra, modelo)

        function treinar_fase!(X_mat, Y_mat, M_mat, epocas::Int, tam_lote::Int, lr::Float32)
            Optimisers.adjust!(estado_otim, lr)
            loader = DataLoader((X_mat, Y_mat, M_mat), batchsize=tam_lote, shuffle=true)
            hist_ppl = Vector{Float32}(undef, epocas)
            Flux.trainmode!(modelo)
            for ep in 1:epocas
                soma_ce, n_b = 0.0f0, 0
                for (xb, yb, mb) in loader
                    y_oh = onehotbatch(vec(yb), 1:V)
                    pesos = reshape(mb, 1, :)
                    denom = max(1.0f0, sum(pesos))

                    perda_ce, (g_slm,) = Flux.withgradient(modelo) do m
                        logits_2d = reshape(m(xb), V, :)
                        ce_por_tok = -sum(y_oh .* Flux.logsoftmax(logits_2d; dims=1); dims=1)
                        sum(ce_por_tok .* pesos) / denom
                    end
                    Flux.update!(estado_otim, modelo, g_slm)
                    soma_ce += perda_ce
                    n_b += 1
                end
                hist_ppl[ep] = exp(soma_ce / n_b)
            end
            return hist_ppl
        end

        ppl_pre = treinar_fase!(X_pre, Y_pre, M_pre, 45, 32, 0.007f0)
        ppl_sft = treinar_fase!(X_sft, Y_sft, M_sft, 75, 10, 0.004f0)
        Flux.testmode!(modelo)

        return (
            modelo=modelo, tok2id=tok2id, id2tok=id2tok, tam_ctx=tam_ctx,
            ppl_pre=ppl_pre, ppl_sft=ppl_sft, freq_zipf=freq_zipf, palavras_ord=palavras_ord
        )
    end

    artefatos_slm = construir_e_treinar_slm();
    md"✅ **SLM Machadiana treinada na CPU!** Vocabulário: `$(length(artefatos_slm.id2tok))` tokens | Perplexidade final Pré-Treino: `$(round(last(artefatos_slm.ppl_pre), digits=2))` | Perplexidade final SFT: `$(round(last(artefatos_slm.ppl_sft), digits=2))`."
end

# ╔═╡ 10000000-0000-4000-8000-000000000028
let
    p_zipf = plot(
        1:length(artefatos_slm.freq_zipf), artefatos_slm.freq_zipf,
        xscale=:log10, yscale=:log10, lw=2.5, marker=:circle, ms=3, color=:teal,
        title="Lei de Zipf no Corpus Machadiano (log-log)",
        xlabel="Posto (Rank) da Palavra", ylabel="Frequência", legend=false
    )
    p_ppl = plot(
        artefatos_slm.ppl_pre, yscale=:log10, lw=2.5, color=:royalblue, label="Fase 1: Pré-Treino (Prosa)",
        title="Queda da Perplexidade (PPL = exp(CE))",
        xlabel="Época", ylabel="Perplexidade (escala log)"
    )
    plot!(p_ppl, artefatos_slm.ppl_sft, yscale=:log10, lw=2.5, color=:crimson, label="Fase 2: Instruction SFT")
    plot(p_zipf, p_ppl, layout=(1, 2), size=(880, 320))
end

# ╔═╡ 10000000-0000-4000-8000-000000000029
md"""
### 🖋️ Converse com a Mini-SLM Machadiana (Inferência Instantânea em CPU)

- **Escolha um tema literário:** $(@bind prompt_opcoes PlutoUI.Select(["como eram os olhos de capitu?" => "Dom Casmurro: Os olhos de Capitu", "qual é a filosofia do humanitismo e das batatas?" => "Quincas Borba: O Humanitismo e as batatas", "qual é a fronteira entre a loucura e a razão?" => "O Alienista: A loucura e a Casa Verde", "o que buscava brás cubas em suas memórias?" => "Memórias Póstumas: O emplasto e a vaidade", "como o tempo trata os homens e a memória?" => "Crônicas e Contos: O tecido invisível do tempo", "qual conselho deixa ao leitor sobre ideias fixas?" => "Aforismos: Ideias fixas e ironia"]))
- **Ou escreva sua pergunta livre:** $(@bind prompt_livre PlutoUI.TextField(; default=""))
- **Temperatura (``\tau``):** $(@bind temp_slm PlutoUI.Slider(0.05:0.05:1.30, default=0.35, show_value=true))
- **Amostragem Top-``k``:** $(@bind topk_slm PlutoUI.Slider(1:1:15, default=4, show_value=true))
- **Semente pseudoaleatória:** $(@bind seed_slm PlutoUI.Slider(1:1:50, default=7, show_value=true))
"""


# ╔═╡ 10000000-0000-4000-8000-000000000030
let
    pergunta_ativa = isempty(strip(prompt_livre)) ? prompt_opcoes : strip(prompt_livre)
    tok2id = artefatos_slm.tok2id
    id2tok = artefatos_slm.id2tok
    tam_ctx = artefatos_slm.tam_ctx
    modelo = artefatos_slm.modelo

    id_pad = tok2id["<pad>"]
    id_unk = tok2id["<unk>"]
    id_fim = tok2id["<fim>"]

    toks_entrada = [w for w in tokenizar_machado(pergunta_ativa) if haskey(tok2id, w)]
    isempty(toks_entrada) && (toks_entrada = ["caro", "leitor", "?"])
    contexto_ids = [tok2id[w] for w in vcat(["<pergunta>"], toks_entrada, ["<machado>"])]

    rng = MersenneTwister(seed_slm)
    gerados = String[]
    top5_inicial = Tuple{String, Float32}[]

    for passo in 1:36
        janela = contexto_ids[max(1, end - tam_ctx + 1):end]
        mat_in = reshape(janela, :, 1)
        logits_passo = copy(vec(modelo(mat_in)[:, end, begin]))

        logits_passo[id_pad] = -1.0f9
        logits_passo[id_unk] = -1.0f9
        logits_passo[tok2id["<pergunta>"]] = -1.0f9
        logits_passo[tok2id["<machado>"]]  = -1.0f9

        if passo == 1
            probs_completas = softmax(logits_passo ./ Float32(max(temp_slm, 0.05)))
            idx5 = partialsortperm(probs_completas, 1:5; rev=true)
            top5_inicial = [(id2tok[i], probs_completas[i]) for i in idx5]
        end

        proximo_id = if temp_slm <= 0.05
            argmax(logits_passo)
        else
            k_ef = clamp(topk_slm, 1, length(logits_passo))
            idx_k = partialsortperm(logits_passo, 1:k_ef; rev=true)
            probs_k = softmax(logits_passo[idx_k] ./ Float32(temp_slm))
            u = rand(rng, Float32)
            acum = 0.0f0
            esc = last(idx_k)
            for (i, p) in enumerate(probs_k)
                acum += p
                if u <= acum
                    esc = idx_k[i]
                    break
                end
            end
            esc
        end

        proximo_id == id_fim && break
        push!(contexto_ids, proximo_id)
        push!(gerados, id2tok[proximo_id])
    end

    resposta_formatada = detokenizar_machado(gerados)
    resumo_top5 = join(["`$(w)` ($(round(p * 100, digits=1))%)" for (w, p) in top5_inicial], " · ")

    md"""
    **Pergunta enviada:** *\"$(pergunta_ativa)\"* *(tokens reconhecidos no léxico: `$(join(toks_entrada, ", "))`)*

    > **Resposta da SLM Machadiana:**
    > *\"$(resposta_formatada)\"*

    **🔍 Top-5 candidatos probabilísticos para o 1º token gerado (``\tau = $(temp_slm)``):** $(resumo_top5)
    """
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Flux = "~0.16.15"
MLUtils = "~0.4.15"
OneHotArrays = "~0.2.11"
Optimisers = "~0.4.11"
Plots = "~1.41.8"
PlutoUI = "~0.7.84"
Zygote = "~0.7.18"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "6af0dbc4bf19f7f079d74f2e98b6dd5c6fe30f63"

[[deps.ADTypes]]
deps = ["PrecompileTools"]
git-tree-sha1 = "629de23e1c16911b439dabd2303c08af9575b226"
registries = "General"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.24.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
registries = "General"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AcceleratedKernels]]
deps = ["Adapt", "ArgCheck", "Atomix", "GPUArraysCore", "KernelAbstractions", "PrecompileTools"]
git-tree-sha1 = "2aa85cbb80ee258e8d3932f6377fe5406f6c7768"
registries = "General"
uuid = "6a4ca0a5-0e36-4168-a932-d9be78d558f1"
version = "0.5.13"

    [deps.AcceleratedKernels.extensions]
    AMDGPUExt = "AMDGPU"
    CUDACoreExt = "CUDACore"
    MetalExt = "Metal"
    OpenCLExt = "OpenCL"
    oneAPIExt = "oneAPI"

    [deps.AcceleratedKernels.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "8c0bc12397dabf5d2dfc4b5763ec463854cc44d3"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.46"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "eec8a0ad4e4b673afba213931504c360fffb8f97"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.8.0"

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
registries = "General"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "0aa371c875e24cf773c87b2f50ab03e4545748af"
registries = "General"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.7.0"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "59ba860cf4d7e2fe58f35c8d08cfe765b6c33655"
registries = "General"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.5"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "cceb62468025be98d42a5dc581b163c20896b040"
registries = "General"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.9"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
registries = "General"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0cfb779e24fdced77a05df1dbe99200821a9f563"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays"]
git-tree-sha1 = "f3cdc9ecdea0c528f7d9e0b031b0d7ae0aa2bf3c"
registries = "General"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.74.13"

[[deps.ChainRulesCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "295fa9b898db109bc56df62c0fc20bd105cec00e"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.8"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkSplitters]]
git-tree-sha1 = "1c52c8e2673edc030191177ff1aee42d25149acb"
registries = "General"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.2.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
registries = "General"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "060adaf7a5c2e14bca025edf0eea0206ffdebd87"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.32.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "af523e59de94de05b9b367a2ca4d63233aacd222"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.5"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "1b26bfe1e23b0fbf350ef5e37a8608caeeafd9de"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.3"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "0737a3e59f9d94957f5b177fa2ca40901636b85b"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
registries = "General"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerCaching]]
deps = ["LMDB", "SHA", "Scratch"]
git-tree-sha1 = "1de3647e392372f1a77e41ae705ceaa12473bc08"
registries = "General"
uuid = "9db33cc3-5358-4881-8759-fa4194144afd"
version = "0.5.3"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b1bc10667320dd5367aff450a992c951ea38fbbc"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.1"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
registries = "General"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
registries = "General"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "6a4a3d254ca35fe9b35e456c66589228514d6a01"
registries = "General"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.18.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "38ef52f18cbdfa783ec6c25acc23fb1e8525ec16"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.10.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnzymeCore]]
git-tree-sha1 = "d88c803e26799193db42e47f6e1e28eeb0322e4d"
registries = "General"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.27"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
registries = "General"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f2659a735f9360fedfa29925812380f322679e93"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.10.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
registries = "General"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "7bd13840b4148949e290748071bbe6826b3eabcd"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "889174c6aef868dc0e5f083dd6d94b4d3b786d7f"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.19.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Flux]]
deps = ["ADTypes", "Adapt", "BFloat16s", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "GPUArrays", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "PrecompileTools", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "31303a34576e135792525f71e7e551cf466c24a4"
registries = "General"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.15"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxEnzymeExt = "Enzyme"
    FluxFiniteDifferencesExt = "FiniteDifferences"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]
    FluxMooncakeExt = "Mooncake"
    FluxReactantExt = "Reactant"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "e1e9ba3a8f5d71e6ae698f138a1362aeabb42e9f"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.18.3+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a6fec1fb7a2cebfa203852419f4f7bdd3f9a8b38"
registries = "General"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.5.0"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d95a7a3a7e0f2fffc6291bd33035830aab01310b"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.15.0+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e894be022a2eba58463939f1bf1f7f7006f27f52"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.19+0"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "416b521e1158dd93cd99b3adcd872085726b36a1"
registries = "General"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
registries = "General"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GPUArrays]]
deps = ["AcceleratedKernels", "Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "1cd2703c11e873a60d0182b6806217a26d71d252"
registries = "General"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "12.2.0"

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

    [deps.GPUArrays.weakdeps]
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "f60436514e6d73f8ee069ae4ee0ff038fe96b69d"
registries = "General"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.2"

[[deps.GPUCompiler]]
deps = ["CompilerCaching", "ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "REPL", "ScopedValues", "TOML", "Tracy", "UUIDs", "tree_sitter_gcn_jll", "tree_sitter_llvm_jll", "tree_sitter_ptx_jll", "tree_sitter_spirv_jll"]
git-tree-sha1 = "2a66558ca5eddcdaa1a25ffb307c9e12f5df8028"
registries = "General"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "2.22.2"

    [deps.GPUCompiler.extensions]
    CthulhuExt = "Cthulhu"
    HighlightsExt = "Highlights"

    [deps.GPUCompiler.weakdeps]
    AMDGPU_LLVM_Backend_jll = "cc5c0156-bd05-5a77-8a68-bb0aafb29019"
    Cthulhu = "f68482b8-f384-11e8-15f7-abe071a5a75f"
    Highlights = "eafb193a-b7ab-5a9e-9068-77385905fa72"
    LLVMDowngrader_jll = "f52de702-fb25-5922-94ba-81dd59b07444"
    NVPTX_LLVM_Backend_jll = "ef6e0fe3-e6ef-59c0-bde6-4989574699e0"
    SPIRV_LLVM_Backend_jll = "4376b9bf-cff8-51b6-bb48-39421dff0d0c"
    SPIRV_LLVM_Translator_jll = "4a5d46fc-d8cf-5151-a261-86b458210efb"

[[deps.GPUToolbox]]
git-tree-sha1 = "5d825bab8a72d96e350e47e30616cf6a4b02fb49"
registries = "General"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "3.6.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
registries = "General"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
registries = "General"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "ea9f1fb37ff10597353b56f3ba0ebd525acef6e0"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.90.1+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "59d266e426ab9695f625fe2b6a51c2d3a04f1ecd"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.15001.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
registries = "General"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "550c21420b96b46f2f0abe8fc4c9430cd55649e3"
registries = "General"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.15.0+3"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "eb7dbaba9b60b122c6526b369291695d25e4128a"
registries = "General"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.25"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
registries = "General"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "d22e8bdb3f03c300a103ff23c0dd5a654c6701de"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.7"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
registries = "General"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "06d90f699285d055e34f0b014636a75a17ff2252"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.13.4"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]
    JSONSerializationExt = ["Serialization"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e06b451acfb1c6fa5ea753b5042207226271f455"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "GPUCompiler", "GPUToolbox", "KernelInterface", "LLVM", "MacroTools", "PrecompileTools", "Printf", "Random", "Random123", "RandomNumbers", "SPIRVIntrinsics", "SPIRV_LLVM_Backend_jll", "SPIRV_Tools_jll", "StaticArraysCore", "ThreadTeams", "UnsafeAtomics", "pocl_standalone_jll"]
git-tree-sha1 = "08b081d7d69e3cc989babee2c74c1d16c0f6baa1"
registries = "General"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.10.7"

    [deps.KernelAbstractions.extensions]
    AMDGPUExt = "AMDGPU"
    LinearAlgebraExt = "LinearAlgebra"
    MetalExt = "Metal"
    SparseArraysExt = "SparseArrays"
    StaticArraysExt = "StaticArrays"
    oneAPIExt = "oneAPI"

    [deps.KernelAbstractions.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.KernelInterface]]
git-tree-sha1 = "626abe5ae4c64c2e5600078962d67bbfab983ce3"
registries = "General"
uuid = "4ee993da-d684-4d17-a7dd-4e58e78d92bf"
version = "0.4.6"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d7c65411dda4bfa0beb7490476122b4f805cf31"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.3.0+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "ScopedValues", "Unicode"]
git-tree-sha1 = "41c2ce7ec5a14c89b93dc6a5853bf2a91d7a8583"
registries = "General"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "10.4.3"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "39def0b4521e0ef2bf26960d8965729f496bb4ec"
registries = "General"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.56+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LMDB]]
deps = ["CEnum", "LMDB_jll"]
git-tree-sha1 = "84e5cba87693714bfd67f589a91f24a996ff3e69"
registries = "General"
uuid = "11f193de-5e89-5f17-923a-7d207d56daf9"
version = "3.0.1"

[[deps.LMDB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "196c962b4d66b978b9be1e15e4e1d53441d5f028"
registries = "General"
uuid = "6206cf0b-f360-5984-af49-5437264c140e"
version = "1.0.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
registries = "General"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "52847246e86dd3dd88c39716748c57fd2bc43dce"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.6"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "c4ab44fe709638fda6f2c0cbfea2c114932d6c2f"
registries = "General"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.1.0"

    [deps.MLCore.extensions]
    MLCorePythonCallExt = "PythonCall"

    [deps.MLCore.weakdeps]
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "acac644e9ca46c512b843723bb55fbed9cc0fccb"
registries = "General"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.18.1"

    [deps.MLDataDevices.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    ChainRulesExt = "ChainRules"
    ComponentArraysExt = "ComponentArrays"
    FillArraysExt = "FillArrays"
    GPUArraysSparseArraysExt = ["GPUArrays", "SparseArrays"]
    MLUtilsExt = "MLUtils"
    MetalExt = ["GPUArrays", "Metal"]
    OneHotArraysExt = "OneHotArrays"
    OpenCLExt = ["GPUArrays", "OpenCL"]
    ReactantExt = "Reactant"
    RecursiveArrayToolsExt = "RecursiveArrayTools"
    ReverseDiffExt = "ReverseDiff"
    SparseArraysExt = "SparseArrays"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"
    cuDNNExt = ["CUDA", "cuDNN"]
    oneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "CodeTracking", "Compat", "DataAPI", "DelimitedFiles", "Distributed", "InteractiveUtils", "MLCore", "Mmap", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "81ba0ccf27958ec0d9575375a42228056fe31a94"
registries = "General"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.15"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "d9d20d6eb2a3290ba5c70a1c013b45305cc39c9e"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.10.20"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "PrecompileTools", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "cffb82e73de54644026b7b51b69f44b4c79f6d8d"
registries = "General"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.60"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreCUDNNExt = ["EnzymeCore", "CUDA", "cuDNN"]
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibMooncakeExt = "Mooncake"
    NNlibReactantExt = "Reactant"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "676c0f1c1c34224d6869675047a54b42fdc7b8c5"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "ScopedValues", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "a73a90d87e3f941f5084d34d56934c2fb59dd9b1"
registries = "General"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.9"

    [deps.OhMyThreads.extensions]
    MarkdownExt = "Markdown"
    ProgressMeterExt = "ProgressMeter"

    [deps.OhMyThreads.weakdeps]
    Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
    ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "9510d7008275fc5b33fc72a73f8fddef0b5430c6"
registries = "General"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.11"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Compat", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "ce02e5e508795e3df43be2f6a2220e3fd0aae206"
registries = "General"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.11"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"
    OptimisersReactantMLDataDevicesExt = ["Reactant", "MLDataDevices"]

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.47.0+0"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "e5ae52d1f4f19a9591d615cb0f1ef8d415b40933"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
registries = "General"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "870db50cd0b5bd7a2f083512cf4b0f1e790453c2"
registries = "General"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.8"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "9869376035f47756b8b8e711fd0d06a06d51da6f"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.84"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "1e697ad6f68af89cd3651bbe9b4a7b5d800adf5a"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.4.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
registries = "General"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
registries = "General"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
registries = "General"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
registries = "General"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
registries = "General"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
registries = "General"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
registries = "General"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
registries = "General"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
registries = "General"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0c0dcffc464cda2ff33fc8ffc1dd1bdd53cd98be"
registries = "General"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.4.0"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
registries = "General"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.SLEEF_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e917bfc9d0aabb490b94bce45c700781ce47c65e"
registries = "General"
uuid = "63e82ce6-3d80-5af4-a84c-484b71ab98bb"
version = "3.9.0+2"

[[deps.SPIRVIntrinsics]]
deps = ["GPUToolbox", "LLVM", "UnsafeAtomics"]
git-tree-sha1 = "2ee2b6f667d53d1f03079ce7179f7de7ef5058b7"
registries = "General"
uuid = "71d1d633-e7e8-4a92-83a1-de8814b09ba8"
version = "1.6.0"

    [deps.SPIRVIntrinsics.extensions]
    SPIRVIntrinsicsSIMDExt = "SIMD"
    SPIRVIntrinsicsSIMDSpecialFunctionsExt = ["SIMD", "SpecialFunctions"]
    SPIRVIntrinsicsSpecialFunctionsExt = "SpecialFunctions"

    [deps.SPIRVIntrinsics.weakdeps]
    GPUCompiler = "61eb1bfa-7361-4325-ad38-22787b887f55"
    SIMD = "fdea26ae-647d-5447-a871-4b548cad5224"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.SPIRV_LLVM_Backend_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b277c67df09ba43963be35adc4acf6b6262a9066"
registries = "General"
uuid = "4376b9bf-cff8-51b6-bb48-39421dff0d0c"
version = "23.1.4+5"

[[deps.SPIRV_Tools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "783c358eff59056d7e703a644f5179270c43226a"
registries = "General"
uuid = "6ac6d60f-d740-5983-97d7-a4482c0689f4"
version = "2025.4.0+0"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
registries = "General"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
registries = "General"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "89880a2cca5cabc68c9023f58e544beda1d2fc1f"
registries = "General"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.2.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
registries = "General"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
registries = "General"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
registries = "General"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "eec446511ab8c3293dd846c61c15128392fceed5"
registries = "General"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.3"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d1a08b3cb5d1470a2b6dc98ba644f1436d5a740f"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.2"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
registries = "General"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StaticArraysCore]]
git-tree-sha1 = "bf7e2d0d9ed9ac2a652fd598c0f784910cc759f2"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.5"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "4daaddbe37aa80b4551bf7c56adf53e7cfcdf35a"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.4"

    [deps.StructArrays.extensions]
    AdaptExt = "Adapt"
    GPUArraysCoreExt = "GPUArraysCore"
    KernelAbstractionsExt = "KernelAbstractions"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"
    StaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d9cfc07e1d1a8985aa9d03870d642b5d7ab2cf33"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.12.1"

    [deps.StructUtils.extensions]
    StructUtilsDBInterfaceExt = ["DBInterface", "Tables"]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DBInterface = "a10d1c49-ce27-4219-8d33-6db1a4562965"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
registries = "General"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadTeams]]
deps = ["ChunkSplitters", "OhMyThreads", "ScopedValues"]
git-tree-sha1 = "15736830cfa2bc55b27769222cefb371800d4f25"
registries = "General"
uuid = "100744f9-5cbb-42a9-938e-f15a20f71997"
version = "1.4.0"
weakdeps = ["EnzymeCore"]

    [deps.ThreadTeams.extensions]
    ThreadTeamsEnzymeCoreExt = "EnzymeCore"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
registries = "General"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "eb3b85ca904a1d1647b15acc091b113b8874b648"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
deps = ["LLVM"]
git-tree-sha1 = "74ac5e0e702972da953ffc981094617070be582d"
registries = "General"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.4.0"
weakdeps = ["GPUCompiler"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsGPUCompilerExt = "GPUCompiler"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
registries = "General"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
registries = "General"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
registries = "General"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b826aedca5fce2e3c91904e338c279e5c1a2da74"
registries = "General"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.15.4+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
registries = "General"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
registries = "General"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "ce91b54a5f90c4f27df6950706234b34e49cf79e"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.14+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
registries = "General"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "704274ca960c42049fd9bda089fc4be32af5cdca"
registries = "General"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.5+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
registries = "General"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
registries = "General"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a54cf54677c9f6eac0d97e9d757c233b56bc21b9"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.13+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
registries = "General"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
registries = "General"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
registries = "General"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
registries = "General"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
registries = "General"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
registries = "General"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
registries = "General"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "9a5e6e8f1a7c40941465c04840145e527f0d6b4b"
registries = "General"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.5.0+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "df77b22eb77051b3d0ef30aab979684f7482c66f"
registries = "General"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.48.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.2+0"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "125b4d8d3459c439714d7476391ee5157958c85a"
registries = "General"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.18"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteCUDAExt = "CUDA"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "c6a86c133861234450ab260dee01b42abd604095"
registries = "General"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.8"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
registries = "General"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
registries = "General"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "21e05cef1411bfe6b22a1ca8cbdc64a297d4b447"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
registries = "General"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
registries = "General"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
registries = "General"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "21cbc8fc2ac0bb182ed461e556d9c062ae6a3164"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.24.1+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
registries = "General"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.70.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.3+0"

[[deps.pocl_standalone_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "Libdl", "SLEEF_jll", "Zstd_jll"]
git-tree-sha1 = "305f3b390a68ab4297626cbe7eeaea88869e2cc0"
registries = "General"
uuid = "54f56a70-6062-5590-a942-1226658f6c83"
version = "7.2.3+8"

[[deps.tree_sitter_gcn_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7aea9f731552967bda93cb7cbf6925de65bc38d5"
registries = "General"
uuid = "8b5cbfcf-8811-596e-8e87-0e51e05ee4b2"
version = "0.1.0+0"

[[deps.tree_sitter_llvm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2fb7d0e3a8abf532fa922766ca130bf814df1e70"
registries = "General"
uuid = "44208993-ee63-5069-9443-8e43b04a9b30"
version = "1.1.0+0"

[[deps.tree_sitter_ptx_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e6ad264eebe3e34f03bca6b5e811d80e0f4797a2"
registries = "General"
uuid = "71e3f6e6-c059-5e7c-a2f2-d560fdad7ce5"
version = "0.1.0+0"

[[deps.tree_sitter_spirv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d8c03949bab84b23fe0500d89979e786d54a653"
registries = "General"
uuid = "f0e86581-c468-54df-a4de-3266e11a3c86"
version = "0.1.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
registries = "General"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─10000000-0000-4000-8000-000000000001
# ╠═10000000-0000-4000-8000-000000000002
# ╟─10000000-0000-4000-8000-000000000003
# ╟─10000000-0000-4000-8000-000000000004
# ╠═10000000-0000-4000-8000-000000000005
# ╟─10000000-0000-4000-8000-000000000006
# ╠═10000000-0000-4000-8000-000000000007
# ╟─10000000-0000-4000-8000-000000000008
# ╠═10000000-0000-4000-8000-000000000009
# ╠═10000000-0000-4000-8000-000000000010
# ╟─10000000-0000-4000-8000-000000000011
# ╠═10000000-0000-4000-8000-000000000012
# ╟─10000000-0000-4000-8000-000000000013
# ╠═10000000-0000-4000-8000-000000000014
# ╟─10000000-0000-4000-8000-000000000015
# ╠═10000000-0000-4000-8000-000000000016
# ╠═10000000-0000-4000-8000-000000000017
# ╟─10000000-0000-4000-8000-000000000018
# ╠═10000000-0000-4000-8000-000000000019
# ╟─10000000-0000-4000-8000-000000000020
# ╟─10000000-0000-4000-8000-000000000021
# ╠═10000000-0000-4000-8000-000000000022
# ╟─10000000-0000-4000-8000-000000000023
# ╠═10000000-0000-4000-8000-000000000024
# ╠═10000000-0000-4000-8000-000000000025
# ╟─10000000-0000-4000-8000-000000000026
# ╠═10000000-0000-4000-8000-000000000027
# ╠═10000000-0000-4000-8000-000000000028
# ╠═10000000-0000-4000-8000-000000000029
# ╠═10000000-0000-4000-8000-000000000030
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
