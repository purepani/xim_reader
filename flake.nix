{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
	crane.url = "github:ipetkov/crane";
    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:adisbladis/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, fenix, crane, pyproject-nix, pyproject-build-systems, uv2nix }:
    let
      system = "x86_64-linux";
      inherit (nixpkgs) lib;
      pkgs = import nixpkgs
        {
          inherit system;
	  overlays = [
	  	fenix.overlays.default
	  ];
        };

	craneLib = (crane.mkLib pkgs).overrideToolchain (p: p.fenix.stable.toolchain);
	src = ./.;
	commonArgs = {
		inherit src;
	};

        craneLibLLvmTools = craneLib.overrideToolchain
          (fenix.packages.${system}.complete.withComponents [
            "cargo"
            "llvm-tools"
            "rustc"
          ]);
	cargoArtifacts = craneLib.buildDepsOnly commonArgs;
	xim_reader = craneLib.buildPackage (
		commonArgs // {
			inherit cargoArtifacts;
		}
	);


      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };
      toolchain = fenix.packages.${system}.stable.toolchain;
      python = pkgs.python313Full;
      rustPlatform = pkgs.makeRustPlatform {
        cargo = toolchain;
        rustc = toolchain;
      };

      pyprojectOverrides = _final: _prev: { };
      pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
        inherit python;
      }).overrideScope
        (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlay
            pyprojectOverrides
          ]
        );
    in
    {
      devShells.${system} = {
        #default = pkgs.mkShell {
        #packages = with pkgs; [
        #bacon
        #fenix.packages.${system}.stable.toolchain
        #python311Full
        #uv
        #];
        #UV_PYTHON = "${pkgs.python311Full}";
        #};
        impure = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
	    pkgs.maturin

            pkgs.bacon
	    toolchain
          ];
        };
        default =
          let
            # Create an overlay enabling editable mode for all local dependencies.
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              # Use environment variable
              root = "$REPO_ROOT";
              # Optional: Only enable editable for these packages
              # members = [ "hello-world" ];
            };
            overrides = _final: _prev: {
              xim_reader = _prev.xim_reader.overrideAttrs (old: {
                nativeBuildInputs = old.nativeBuildInputs ++ [ rustPlatform.cargoSetupHook rustPlatform.maturinBuildHook ];
                cargoDeps = old.src;
              });
            };

            # Override previous set with our overrideable overlay.
            editablePythonSet = pythonSet.overrideScope (lib.composeExtensions editableOverlay overrides);

            # Build virtual environment, with local packages being editable.
            #
            # Enable all optional dependencies for development.
            virtualenv = editablePythonSet.mkVirtualEnv "xim-reader-env" workspace.deps.all;

          in
          craneLib.devShell {
            packages = [
              virtualenv
              pkgs.bacon
              #fenix.packages.${system}.stable.toolchain
              pkgs.uv
            ];
            shellHook = ''
              # Undo dependency propagation by nixpkgs.
              unset PYTHONPATH
              # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
	    #RUST_PATH = "${toolchain}";
	    #RUST_DOC_PATH = "${toolchain}/share/doc/rust/html/std/index.html";
          };
      };


    };
}
