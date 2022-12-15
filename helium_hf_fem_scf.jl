module Helium_HF_FEM_SCF
    include("helium_hf_fem_eigen.jl")
    include("helium_vh_fem.jl")
    using Printf
    using .Helium_HF_FEM_Eigen
    using .Helium_Vh_FEM

    const DR = 0.01
    const MAXITER = 100
    const THRESHOLD = 1.0E-20

    function scfloop()
        # Schrödinger方程式を解くルーチンの初期処理
        hfem_param, hfem_val = Helium_HF_FEM_Eigen.construct()

        # 有限要素法のデータのみ生成
        Helium_HF_FEM_Eigen.make_wavefunction(0, hfem_param, hfem_val, nothing)
        
        # Poisson方程式を解くルーチンの初期処理
        vh_param, vh_val = Helium_Vh_FEM.construct(hfem_param)
        
        # 仮の電子密度でPoisson方程式を解く
        Helium_Vh_FEM.solvepoisson!(0, hfem_param, hfem_val, vh_val)

        # 新しく計算されたエネルギー
        enew = 0.0

        # CPUのノルムの和を初期化
        sum_norm_cpu = 0.0

        # GPUのノルムの和を初期化
        sum_norm_gpu = 0.0

        for iter in 1:MAXITER
            eigenenergy, norm_cpu, norm_gpu = Helium_HF_FEM_Eigen.make_wavefunction(iter, hfem_param, hfem_val, vh_val)
            sum_norm_cpu += norm_cpu
            sum_norm_gpu += norm_gpu
            
            # 前回のSCF計算のエネルギーを保管
            eold = enew
            
            # 今回のSCF計算のエネルギーを計算する
            enew = Helium_HF_FEM_Eigen.get_totalenergy(eigenenergy, hfem_param, hfem_val, vh_val)

            @printf "Iteration # %2d:\n" iter
            
            # 今回のSCF計算のエネルギーと前回のSCF計算のエネルギーの差の絶対値
            ediff = abs(enew - eold)

            # SCF計算が収束したかどうか
            if ediff <= THRESHOLD
                # 波動関数を規格化
                hfem_val.phi ./= sqrt(4.0 * pi)

                @printf "CPUのノルムの和 = %.16f, GPUのノルムの和 = %.16f\n" sum_norm_cpu sum_norm_gpu

                # 収束したのでhfem_param, hfem_val, エネルギーを返す
                return hfem_param, hfem_val, enew
            end
            
            # Poisson方程式を解く
            Helium_Vh_FEM.solvepoisson!(scfloop, hfem_param, hfem_val, vh_val)
        end

        @printf "CPUのノルムの和 = %.16e, GPUのノルムの和 = %.16e\n" sum_norm_cpu sum_norm_gpu

        return nothing
    end

    save_result(hfem_param, hfem_val, filename) = let
        open(filename, "w" ) do fp
            len = length(hfem_val.node_r_glo)
            imax = floor(Int, (hfem_val.node_r_glo[len] - hfem_val.node_r_glo[1]) / DR)
            for i = 0:imax
                r = hfem_val.node_r_glo[1] + DR * float(i)
                println(fp, @sprintf "%.14f, %.14f" r Helium_Vh_FEM.phi(hfem_param, hfem_val, r))
            end
        end
    end
end

