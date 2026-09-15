{
  description = "GS-CBBA multi-harvester simulation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?rev=4c1018dae018162ec878d42fec712642d214fdfa";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        metadata = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project;
        dependencies = with python.pkgs; [ numpy scipy matplotlib seaborn pulp ];
        package = python.pkgs.buildPythonPackage {
          pname = metadata.name;
          version = metadata.version;
          pyproject = true;
          src = pkgs.lib.cleanSource ./.;
          build-system = [ python.pkgs.hatchling ];
          inherit dependencies;
          nativeCheckInputs = [ python.pkgs.pytestCheckHook ];
          pythonImportsCheck = [ "gscbba" "gscbba.cbba" "gscbba.baselines" ];
          meta.mainProgram = "gscbba";
        };
        environment = python.withPackages (ps: dependencies ++ [
          package ps.build ps.hatchling ps.pytest
        ]);
      in {
        packages.default = package;
        apps.default = {
          type = "app";
          program = "${package}/bin/gscbba";
        };
        devShells.default = pkgs.mkShell {
          packages = [ environment pkgs.ruff ];
          MPLBACKEND = "Agg";
          shellHook = ''
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
          '';
        };
        checks.default = package;
      });
}
