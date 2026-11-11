import pandas as pd
from alpha_edge.market.aea87_identity_forensics import plot_case, normalize


def test_plot_generates_png_bytes_without_local_file(tmp_path):
    p = pd.DataFrame(
        {
            "asset_id": ["CRYPTO:HYPE-USD"] * 3,
            "date": pd.to_datetime(["2024-11-28", "2024-11-29", "2024-11-30"]),
            "close_adjusted_usd": [0.01, 6.0, 7.0],
        }
    )
    flags, frame = normalize(p, pd.DataFrame(), "CRYPTO:HYPE-USD", inception="2024-11-29")
    img = plot_case(frame, flags, None, inception="2024-11-29")
    assert img.startswith(bytes.fromhex("89504e470d0a1a0a"))
    assert list(tmp_path.iterdir()) == []
