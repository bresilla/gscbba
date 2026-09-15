local make = oslo.make

make.recipe{
  name = "setup",
  desc = "check the Python environment",
  run = function()
    sh.python("-c", "import numpy, scipy, matplotlib, seaborn, pulp, gscbba")
    sh.ruff("--version")
  end,
}

make.recipe{
  name = "build",
  desc = "build a wheel and source archive in dist/",
  deps = { "setup" },
  run = function()
    sh.python("-m", "build", "--no-isolation", "--outdir", "dist")
  end,
}
make.alias("b", "build")

make.recipe{
  name = "run",
  desc = "run the simulator; pass flags through --args",
  deps = { "setup" },
  params = { { "--args", desc = "CLI arguments" } },
  run = function(a)
    local argv = { "-m", "gscbba" }
    for _, word in ipairs(a.rest or {}) do argv[#argv + 1] = word end
    if type(a.args) == "string" then
      for word in a.args:gmatch("%S+") do argv[#argv + 1] = word end
    end
    sh.python(table.unpack(argv))
  end,
}
make.alias("r", "run")

make.recipe{
  name = "test",
  desc = "run the test suite",
  deps = { "setup" },
  run = function() sh.python("-m", "pytest") end,
}
make.alias("t", "test")

make.recipe{
  name = "fmt",
  desc = "format Python files and sort imports",
  run = function()
    sh.ruff("check", "--fix", ".")
    sh.ruff("format", ".")
  end,
}

make.recipe{
  name = "check",
  desc = "check Python formatting and lint",
  run = function()
    sh.ruff("format", "--check", ".")
    sh.ruff("check", ".")
  end,
}

make.recipe{
  name = "verify",
  desc = "run formatting checks and tests",
  deps = { "check", "test" },
}
make.alias("v", "verify")

make.recipe{
  name = "plots",
  desc = "regenerate plots from cached results",
  deps = { "setup" },
  run = function() sh.python("-m", "gscbba", "--figures-only") end,
}
