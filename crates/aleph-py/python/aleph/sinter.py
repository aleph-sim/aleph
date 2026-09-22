"""sinter adapters for aleph's decoders.

    import sinter, aleph.sinter
    sinter.collect(tasks=..., decoders=["aleph-relay-bp-osd"],
                   custom_decoders=aleph.sinter.decoders(), num_workers=8)

Adapters hold only a decoder name and parameters, so they pickle cleanly into
sinter's worker processes; the Rust decoder is built per DEM in each worker.
"""
from __future__ import annotations

try:
    import sinter as _sinter
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "aleph.sinter requires sinter; install with: pip install 'aleph-sim[sinter]'"
    ) from e

from . import qec as _qec

__all__ = ["AlephSinterDecoder", "CompiledAlephDecoder", "decoders"]


class CompiledAlephDecoder(_sinter.CompiledDecoder):
    """An aleph decoder built for one DEM."""

    def __init__(self, decoder: _qec.Decoder) -> None:
        self._decoder = decoder

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        return self._decoder.decode_batch_bit_packed(bit_packed_detection_event_data)


class AlephSinterDecoder(_sinter.Decoder):
    """sinter decoder for aleph decoder ``name`` with keyword ``params``."""

    def __init__(self, name: str, **params) -> None:
        self.name = name
        self.params = dict(params)

    def compile_decoder_for_dem(self, *, dem) -> CompiledAlephDecoder:
        model = _qec.DetectorErrorModel(dem)
        return CompiledAlephDecoder(_qec.Decoder(model, self.name, **self.params))

    def decode_via_files(self, *, num_shots, num_dets, num_obs, dem_path,
                         dets_b8_in_path, obs_predictions_b8_out_path, tmp_dir) -> None:
        import stim

        dem = stim.DetectorErrorModel.from_file(dem_path)
        dets = stim.read_shot_data_file(path=dets_b8_in_path, format="b8",
                                        num_detectors=num_dets, bit_packed=True)
        pred = self.compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(
            bit_packed_detection_event_data=dets)
        stim.write_shot_data_file(data=pred, path=obs_predictions_b8_out_path, format="b8",
                                  num_observables=num_obs)

    def __repr__(self) -> str:
        return f"AlephSinterDecoder({self.name!r}, **{self.params!r})"


def decoders(params: dict | None = None) -> dict[str, AlephSinterDecoder]:
    """``{"aleph-<name>": AlephSinterDecoder}`` for every aleph decoder.

    ``params`` maps a decoder name (e.g. ``"relay-bp"``) to its keyword parameters.
    """
    params = params or {}
    return {f"aleph-{n}": AlephSinterDecoder(n, **params.get(n, {})) for n in _qec.decoder_names()}
