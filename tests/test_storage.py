"""Comprehensive unit tests for faceauth.storage module.

Tests the EmbeddingStore class which handles saving, loading, and managing
face embeddings as .npz files with appropriate security permissions.
"""

import os
import stat
from pathlib import Path

import numpy as np
import pytest

from faceauth.storage import EmbeddingStore


def create_embedding(seed: int = 42, dim: int = 512) -> np.ndarray:
    """Create a random 512-dimensional embedding for testing."""
    rng = np.random.default_rng(seed)
    return rng.random(dim, dtype=np.float32)


@pytest.mark.unit
def test_save_and_load_single_embedding(tmp_path):
    """Test saving and loading a single embedding roundtrip."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "testuser"

    # Create and save embedding
    embedding = create_embedding(seed=1)
    saved_path = store.save(username, [embedding])

    # Verify file was created
    assert saved_path.exists()
    assert saved_path == tmp_path / f"{username}.npz"

    # Load and verify
    loaded = store.load(username)
    assert len(loaded) == 1
    assert loaded[0].shape == (512,)
    np.testing.assert_array_almost_equal(loaded[0], embedding)


@pytest.mark.unit
def test_save_and_load_multiple_embeddings(tmp_path):
    """Test saving and loading multiple embeddings with correct shape and values."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "multiuser"

    # Create multiple embeddings
    embeddings = [
        create_embedding(seed=1),
        create_embedding(seed=2),
        create_embedding(seed=3),
    ]

    store.save(username, embeddings)
    loaded = store.load(username)

    # Verify count and shapes
    assert len(loaded) == 3
    for emb in loaded:
        assert emb.shape == (512,)

    # Verify values match
    for original, loaded_emb in zip(embeddings, loaded):
        np.testing.assert_array_almost_equal(loaded_emb, original)


@pytest.mark.unit
def test_load_non_enrolled_user_returns_empty_list(tmp_path):
    """Test loading embeddings for a user who hasn't been enrolled."""
    store = EmbeddingStore(data_dir=tmp_path)

    # Load from non-existent user
    loaded = store.load("nonexistent")
    assert loaded == []
    assert isinstance(loaded, list)


@pytest.mark.unit
def test_delete_enrolled_user_returns_true(tmp_path):
    """Test deleting an enrolled user returns True."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "deleteuser"

    # Enroll user
    embedding = create_embedding(seed=5)
    store.save(username, [embedding])
    assert store.is_enrolled(username)

    # Delete should return True
    result = store.delete(username)
    assert result is True
    assert not store.is_enrolled(username)


@pytest.mark.unit
def test_delete_non_enrolled_user_returns_false(tmp_path):
    """Test deleting a non-enrolled user returns False."""
    store = EmbeddingStore(data_dir=tmp_path)

    # Delete non-existent user should return False
    result = store.delete("nonexistent")
    assert result is False


@pytest.mark.unit
def test_list_users_returns_sorted_usernames(tmp_path):
    """Test list_users returns alphabetically sorted usernames."""
    store = EmbeddingStore(data_dir=tmp_path)

    # Enroll users in non-alphabetical order
    usernames = ["charlie", "alice", "bob"]
    for username in usernames:
        store.save(username, [create_embedding(seed=hash(username) % 1000)])

    # Should return sorted
    users = store.list_users()
    assert users == ["alice", "bob", "charlie"]


@pytest.mark.unit
def test_list_users_empty_when_data_dir_not_exists(tmp_path):
    """Test list_users returns empty list when data_dir doesn't exist."""
    # Use a subdirectory that doesn't exist yet
    non_existent_dir = tmp_path / "does_not_exist"
    store = EmbeddingStore(data_dir=non_existent_dir)

    users = store.list_users()
    assert users == []


@pytest.mark.unit
def test_list_users_empty_when_no_enrollments(tmp_path):
    """Test list_users returns empty list when directory exists but is empty."""
    store = EmbeddingStore(data_dir=tmp_path)
    store._ensure_dir()  # Create the directory

    users = store.list_users()
    assert users == []


@pytest.mark.unit
def test_is_enrolled_returns_true_for_enrolled_user(tmp_path):
    """Test is_enrolled returns True for an enrolled user."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "enrolled"

    store.save(username, [create_embedding(seed=10)])
    assert store.is_enrolled(username) is True


@pytest.mark.unit
def test_is_enrolled_returns_false_for_non_enrolled_user(tmp_path):
    """Test is_enrolled returns False for a non-enrolled user."""
    store = EmbeddingStore(data_dir=tmp_path)
    assert store.is_enrolled("nonexistent") is False


@pytest.mark.unit
def test_get_embedding_count_returns_correct_count(tmp_path):
    """Test get_embedding_count returns the correct number of embeddings."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "countuser"

    # Save 5 embeddings
    embeddings = [create_embedding(seed=i) for i in range(5)]
    store.save(username, embeddings)

    count = store.get_embedding_count(username)
    assert count == 5


@pytest.mark.unit
def test_get_embedding_count_returns_zero_for_non_enrolled(tmp_path):
    """Test get_embedding_count returns 0 for non-enrolled user."""
    store = EmbeddingStore(data_dir=tmp_path)
    count = store.get_embedding_count("nonexistent")
    assert count == 0


@pytest.mark.unit
def test_overwrite_existing_enrollment(tmp_path):
    """Test that saving again overwrites the previous enrollment."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "overwriteuser"

    # First enrollment with 2 embeddings
    first_embeddings = [create_embedding(seed=1), create_embedding(seed=2)]
    store.save(username, first_embeddings)
    assert store.get_embedding_count(username) == 2

    # Second enrollment with 3 different embeddings
    second_embeddings = [
        create_embedding(seed=10),
        create_embedding(seed=20),
        create_embedding(seed=30),
    ]
    store.save(username, second_embeddings)

    # Should have 3 embeddings, not 5
    assert store.get_embedding_count(username) == 3

    # Verify they match the second enrollment
    loaded = store.load(username)
    for original, loaded_emb in zip(second_embeddings, loaded):
        np.testing.assert_array_almost_equal(loaded_emb, original)


@pytest.mark.unit
def test_path_traversal_prevention_forward_slash(tmp_path):
    """Test that path traversal attempts raise ValueError."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "../../../etc/passwd"

    embedding = create_embedding(seed=99)

    # Should raise ValueError due to invalid username format
    with pytest.raises(ValueError, match="Invalid username"):
        store.save(username, [embedding])


@pytest.mark.unit
def test_path_traversal_prevention_null_bytes(tmp_path):
    """Test that null bytes in username raise ValueError."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "user\0name"

    embedding = create_embedding(seed=88)

    # Should raise ValueError due to invalid username format
    with pytest.raises(ValueError, match="Invalid username"):
        store.save(username, [embedding])


@pytest.mark.unit
def test_directory_permissions_are_restrictive(tmp_path):
    """Test that data directory has 0o700 permissions after save."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "permuser"

    # Save triggers directory creation
    store.save(username, [create_embedding(seed=50)])

    # Check directory permissions
    dir_stat = os.stat(tmp_path)
    dir_mode = stat.S_IMODE(dir_stat.st_mode)
    assert dir_mode == 0o700, f"Expected 0o700, got {oct(dir_mode)}"


@pytest.mark.unit
def test_file_permissions_are_restrictive(tmp_path):
    """Test that embedding files have 0o600 permissions after save."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "filepermuser"

    saved_path = store.save(username, [create_embedding(seed=60)])

    # Check file permissions
    file_stat = os.stat(saved_path)
    file_mode = stat.S_IMODE(file_stat.st_mode)
    assert file_mode == 0o600, f"Expected 0o600, got {oct(file_mode)}"


@pytest.mark.unit
def test_multiple_users_stored_independently(tmp_path):
    """Test that multiple users can be stored and retrieved independently."""
    store = EmbeddingStore(data_dir=tmp_path)

    # Enroll three users with different numbers of embeddings
    users_data = {
        "alice": [create_embedding(seed=1), create_embedding(seed=2)],
        "bob": [create_embedding(seed=10)],
        "charlie": [
            create_embedding(seed=20),
            create_embedding(seed=21),
            create_embedding(seed=22),
        ],
    }

    # Save all users
    for username, embeddings in users_data.items():
        store.save(username, embeddings)

    # Verify each user independently
    for username, expected_embeddings in users_data.items():
        loaded = store.load(username)
        assert len(loaded) == len(expected_embeddings)
        for expected, loaded_emb in zip(expected_embeddings, loaded):
            np.testing.assert_array_almost_equal(loaded_emb, expected)

    # Verify list_users shows all three
    assert set(store.list_users()) == {"alice", "bob", "charlie"}


@pytest.mark.unit
def test_embedding_dtype_preserved(tmp_path):
    """Test that float32 dtype is preserved through save/load."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "dtypeuser"

    embedding = create_embedding(seed=77)
    assert embedding.dtype == np.float32

    store.save(username, [embedding])
    loaded = store.load(username)

    assert loaded[0].dtype == np.float32


@pytest.mark.unit
def test_embedding_values_precision(tmp_path):
    """Test that embedding values are preserved with high precision."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "precisionuser"

    # Create embedding with specific values
    rng = np.random.default_rng(seed=123)
    embedding = rng.random(512, dtype=np.float32)

    store.save(username, [embedding])
    loaded = store.load(username)

    # Should be exactly equal for float32
    assert np.array_equal(loaded[0], embedding)

    # Also verify with stricter tolerance
    np.testing.assert_allclose(loaded[0], embedding, rtol=1e-7, atol=0)


@pytest.mark.unit
def test_save_returns_correct_path(tmp_path):
    """Test that save() returns the correct Path object."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "pathuser"

    returned_path = store.save(username, [create_embedding(seed=33)])

    assert isinstance(returned_path, Path)
    assert returned_path == tmp_path / f"{username}.npz"
    assert returned_path.exists()


@pytest.mark.unit
def test_delete_removes_file_completely(tmp_path):
    """Test that delete() completely removes the file from filesystem."""
    store = EmbeddingStore(data_dir=tmp_path)
    username = "removeduser"

    # Enroll and verify file exists
    saved_path = store.save(username, [create_embedding(seed=44)])
    assert saved_path.exists()

    # Delete and verify file is gone
    store.delete(username)
    assert not saved_path.exists()

    # Load should return empty list
    assert store.load(username) == []
