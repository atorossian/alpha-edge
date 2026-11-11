import hashlib

from alpha_edge.market.aea87_archive_evidence import archive, inventory


def test_archive_inventory_and_dry_run(tmp_path):
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots" / "price.png").write_bytes(b"image")
    (tmp_path / "summary.json").write_text('{"assets":1}')
    files = inventory(tmp_path)
    assert [x["path"] for x in files] == ["plots/price.png", "summary.json"]
    assert files[0]["sha256"] == hashlib.sha256(b"image").hexdigest()
    result = archive(tmp_path, "bucket", "eu-west-1", "test", dry_run=True)
    assert result["files"] == files
