target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_rules("c++.openmp")
    
    if is_plat("windows") then
        add_cxflags("/O2", "/arch:AVX2")
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", "-O3", "-march=native")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_rules("c++.openmp")

    if is_plat("windows") then
        add_cxflags("/O2", "/arch:AVX2")
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", "-O3", "-march=native")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()
