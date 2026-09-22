"""QEC decoding: stim-format Detector Error Models and aleph's decoders.

    import aleph.qec as qec
    dem = qec.DetectorErrorModel(stim_circuit.detector_error_model(decompose_errors=True))
    predictions = qec.Decoder(dem, "relay-bp-osd").decode_batch(detection_events)

``decoder_names()`` lists the available decoders. stim is not required.
"""
from . import _native

DetectorErrorModel = _native.qec.DetectorErrorModel
Decoder = _native.qec.Decoder
decoder_names = _native.qec.decoder_names

__all__ = ["DetectorErrorModel", "Decoder", "decoder_names"]
