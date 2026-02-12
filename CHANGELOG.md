# Changelog

All notable changes to this project will be documented in this file. / Todos los cambios notables en este proyecto serán documentados en este archivo.

## [Unreleased]

### English

#### Added
- **Architecture**: Reorganized codebase into logical subpackages (`core`, `io`, `validation`, `analysis`, `internals`, `tooling`) to improve maintainability and navigation.
- **Tooling**: Added `simuci.envcheck` (via `python -m simuci.envcheck`) for verifying environment health, dependencies, and optional security auditing (via `pip-audit`).
- **Security**: Added `security` optional dependency group in `pyproject.toml`.
- **Documentation**: Added `ARCHITECTURE.md` (and ES variant) to explain the new modular structure. Added `README.es.md` for Spanish documentation.

#### Changed
- **Internal API**: Moved modules from root `simuci.*` to specific subpackages (e.g., `simuci.stats` → `simuci.analysis.stats`).
- **Imports**: Updated all internal imports to reflect the new structure.
- **Stats**: Made `scipy.stats.PermutationMethod` import conditional to support older SciPy versions without crashing.

#### Fixed
- **Compatibility**: Ensure `anderson_ksamp` works gracefully even if `PermutationMethod` is unavailable in the installed SciPy version.

---

### Español

#### Agregado
- **Arquitectura**: Reorganización del código base en subpaquetes lógicos (`core`, `io`, `validation`, `analysis`, `internals`, `tooling`) para mejorar mantenibilidad y navegación.
- **Herramientas**: Añadido `simuci.envcheck` (vía `python -m simuci.envcheck`) para verificar salud del entorno, dependencias y auditoría de seguridad opcional (vía `pip-audit`).
- **Seguridad**: Añadido grupo de dependencias opcional `security` en `pyproject.toml`.
- **Documentación**: Añadido `ARCHITECTURE.md` (y variante ES) para explicar la nueva estructura modular. Añadido `README.es.md` con documentación en español.

#### Cambiado
- **API Interna**: Módulos movidos desde la raíz `simuci.*` a subpaquetes específicos (ej. `simuci.stats` → `simuci.analysis.stats`).
- **Imports**: Actualizados todos los imports internos para reflejar la nueva estructura.
- **Stats**: La importación de `scipy.stats.PermutationMethod` ahora es condicional para soportar versiones antiguas de SciPy sin errores.

#### Arreglado
- **Compatibilidad**: Asegurado que `anderson_ksamp` funcione correctamente incluso si `PermutationMethod` no está disponible en la versión instalada de SciPy.
