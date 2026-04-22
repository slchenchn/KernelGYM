from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from drkernel.kernel.scripts.materialize_backend_neutral_data import (
    build_backend_neutral_prompt,
    extract_architecture_code,
    format_prompt_messages_for_first_turn,
    format_prompt_row_for_text,
    format_sampled_prompt_row_for_text,
    materialize_backend_neutral_parquet,
    neutralize_prompt_messages,
    sample_prompt_indices,
    write_backend_neutral_text,
    write_formatted_prompt_sample_text,
)

ROOT = Path(__file__).resolve().parents[1]
ACTUAL_TRITON_PARQUET = ROOT / "drkernel" / "data" / "drkernel-rl-data" / "cuda_llm_rl_thinking_1025.parquet"


def test_extract_architecture_from_triton_prompt_wrapper():
    source = """
You write custom Triton kernels.

You are given the following architecture:
```
import torch

class Model:
    pass
```

Optimize the architecture named Model with custom Triton operators!
"""

    assert extract_architecture_code(source) == "import torch\n\nclass Model:\n    pass"


def test_extract_architecture_from_inline_cuda_prompt_wrapper():
    source = """
You are an expert in PyTorch and CUDA programming.

Now you are given the below PyTorch model:
```python
import torch

class Model:
    pass
```

Please respond with thinking process and your final answer.
"""

    assert extract_architecture_code(source) == "import torch\n\nclass Model:\n    pass"


def test_neutralize_prompt_messages_removes_backend_prompt_text():
    messages = [
        {
            "role": "user",
            "content": """
You write custom Triton kernels.

You are given the following architecture:
```
class Model:
    pass
```

Optimize the architecture named Model with custom Triton operators!
""",
        }
    ]

    neutralized = neutralize_prompt_messages(messages)

    assert neutralized == [
        {
            "role": "user",
            "content": "You are given the following PyTorch model:\n```python\nclass Model:\n    pass\n```",
        }
    ]


def test_neutralize_prompt_messages_rejects_rows_without_architecture_block():
    with pytest.raises(ValueError, match="architecture block"):
        neutralize_prompt_messages([{"role": "user", "content": "No model here."}])


@pytest.mark.parametrize(
    ("row_index", "expected_prompt"),
    [
        (
            0,
            "You are given the following PyTorch model:\n"
            "```python\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "\n"
            "\n"
            "class Model(nn.Module):\n"
            "    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):\n"
            "        super().__init__()\n"
            "        self.conv2d = nn.Conv2d(\n"
            "            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias\n"
            "        )\n"
            "\n"
            "    def forward(self, x):\n"
            "        x = self.conv2d(x)\n"
            "        x = torch.log2(torch.abs(x) + 1e-8)  # log2 with small epsilon to prevent log of zero\n"
            "        return x\n"
            "\n"
            "\n"
            "batch_size = 64\n"
            "in_channels = 16\n"
            "out_channels = 32\n"
            "kernel_size = 3\n"
            "stride = 1\n"
            "padding = 1\n"
            "dilation = 1\n"
            "bias = False\n"
            "\n"
            "\n"
            "def get_inputs():\n"
            "    height, width = 256, 256\n"
            "    return [torch.randn(batch_size, in_channels, height, width)]\n"
            "\n"
            "\n"
            "def get_init_inputs():\n"
            "    return [in_channels, out_channels, kernel_size, stride, padding, dilation, bias]\n"
            "```",
        ),
        (
            1,
            "You are given the following PyTorch model:\n"
            "```python\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "\n"
            "\n"
            "class Model(nn.Module):\n"
            "    def __init__(self, num_classes, input_size, on_value, off_value):\n"
            "        super().__init__()\n"
            "        self.num_classes = num_classes\n"
            "        self.input_size = input_size\n"
            "        self.on_value = on_value\n"
            "        self.off_value = off_value\n"
            "\n"
            "    def forward(self, x):\n"
            "        # Convert the input to one-hot encoding\n"
            "        x_one_hot = F.one_hot(x.long(), num_classes=self.num_classes)\n"
            "        # Apply torch.minimum\n"
            "        x_min = torch.minimum(x_one_hot, torch.tensor(self.on_value))\n"
            "        x_min_off = torch.minimum(x_min, torch.tensor(self.off_value))\n"
            "        return x_min_off\n"
            "\n"
            "\n"
            "num_classes = 10\n"
            "input_size = 64\n"
            "on_value = 0.5\n"
            "off_value = 0.3\n"
            "\n"
            "\n"
            "def get_inputs():\n"
            "    # Assuming the input data is in [0, num_classes-1] range\n"
            "    return [torch.randint(0, num_classes, (input_size,))]\n"
            "\n"
            "\n"
            "def get_init_inputs():\n"
            "    return [num_classes, input_size, on_value, off_value]\n"
            "```",
        ),
        (
            2,
            "You are given the following PyTorch model:\n"
            "```python\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class Model(nn.Module):\n"
            "    def __init__(self, divisor, scale_factor):\n"
            "        super().__init__()\n"
            "        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=\"nearest\")\n"
            "        self.divisor = divisor\n"
            "\n"
            "    def forward(self, x):\n"
            "        x = torch.remainder(x, self.divisor)\n"
            "        x = self.upsample(x)\n"
            "        return x\n"
            "\n"
            "\n"
            "batch_size = 32\n"
            "channels = 64\n"
            "height, width = 64, 64\n"
            "divisor = 3\n"
            "scale_factor = 2.0\n"
            "\n"
            "\n"
            "def get_inputs():\n"
            "    return [torch.randn(batch_size, channels, height, width)]\n"
            "\n"
            "\n"
            "def get_init_inputs():\n"
            "    return [divisor, scale_factor]\n"
            "```",
        ),
    ],
)
def test_neutralize_actual_triton_parquet_rows_match_exact_expected_prompt(row_index, expected_prompt):
    if not ACTUAL_TRITON_PARQUET.exists():
        pytest.skip(f"actual Triton parquet not available: {ACTUAL_TRITON_PARQUET}")

    row = pq.read_table(ACTUAL_TRITON_PARQUET, columns=["prompt"]).slice(row_index, 1).to_pylist()[0]

    assert neutralize_prompt_messages(row["prompt"])[0]["content"] == expected_prompt


def test_build_backend_neutral_prompt_has_no_backend_instruction():
    prompt = build_backend_neutral_prompt("class Model:\n    pass")

    assert prompt == "You are given the following PyTorch model:\n```python\nclass Model:\n    pass\n```"


def test_format_prompt_row_for_text_is_human_readable():
    text = format_prompt_row_for_text(
        7,
        [
            {
                "role": "user",
                "content": "You are given the following PyTorch model:\n```python\nclass Model: pass\n```",
            }
        ],
    )

    assert text == (
        "==================== row 7 ====================\n"
        "--- message 0 role=user ---\n"
        "You are given the following PyTorch model:\n"
        "```python\n"
        "class Model: pass\n"
        "```\n"
    )


def test_write_backend_neutral_text_uses_row_separators(tmp_path):
    output_path = tmp_path / "neutral.txt"
    prompts = [
        [{"role": "user", "content": "PROMPT 0"}],
        [{"role": "user", "content": "PROMPT 1"}],
    ]

    write_backend_neutral_text(output_path, prompts)

    assert output_path.read_text() == (
        "==================== row 0 ====================\n"
        "--- message 0 role=user ---\n"
        "PROMPT 0\n"
        "\n"
        "==================== row 1 ====================\n"
        "--- message 0 role=user ---\n"
        "PROMPT 1\n"
    )


def test_format_prompt_messages_for_first_turn_outputs_complete_formatted_prompt():
    messages = [{"role": "user", "content": "\nPROMPT\n"}]

    formatted = format_prompt_messages_for_first_turn(messages, "INSTRUCTION\n\n{{ problem }}\n\nLet's think.\n")

    assert formatted == [{"role": "user", "content": "INSTRUCTION\n\nPROMPT\n\nLet's think.\n"}]
    assert messages == [{"role": "user", "content": "\nPROMPT\n"}]


def test_format_sampled_prompt_row_for_text_uses_sample_and_source_row_labels():
    text = format_sampled_prompt_row_for_text(
        3,
        17,
        [{"role": "user", "content": "INSTRUCTION\n\nPROMPT"}],
    )

    assert text == (
        "==================== sample 3 source_row 17 ====================\n"
        "--- message 0 role=user ---\n"
        "INSTRUCTION\n"
        "\n"
        "PROMPT\n"
    )


def test_sample_prompt_indices_is_deterministic_and_dataset_ordered():
    assert sample_prompt_indices(row_count=10, sample_size=4, seed=3) == [2, 3, 7, 8]


def test_write_formatted_prompt_sample_text_outputs_full_formatted_prompts(tmp_path):
    neutral_parquet_path = tmp_path / "neutral.parquet"
    sample_output_path = tmp_path / "formatted_sample.txt"
    prompt_config_path = tmp_path / "prompt.yaml"
    prompt_type = pa.list_(
        pa.struct(
            [
                pa.field("content", pa.string()),
                pa.field("role", pa.string()),
            ]
        )
    )
    neutral_prompts = [
        [{"role": "user", "content": "PROMPT 0"}],
        [{"role": "user", "content": "PROMPT 1"}],
    ]
    table = pa.table({"prompt": pa.array(neutral_prompts, type=prompt_type), "ability": ["kernel", "kernel"]})
    pq.write_table(table, neutral_parquet_path)
    prompt_config_path.write_text(
        "per_turn_prompts:\n"
        "  - name: first_turn\n"
        "    template: |\n"
        "      INSTRUCTION\n"
        "\n"
        "      {{ problem }}\n"
        "\n"
        "      Let's think.\n",
        encoding="utf-8",
    )

    sample_indices = write_formatted_prompt_sample_text(
        neutral_parquet_path,
        sample_output_path,
        prompt_config_path,
        sample_size=2,
        seed=0,
    )

    assert sample_indices == [0, 1]
    assert sample_output_path.read_text() == (
        "==================== sample 0 source_row 0 ====================\n"
        "--- message 0 role=user ---\n"
        "INSTRUCTION\n"
        "\n"
        "PROMPT 0\n"
        "\n"
        "Let's think.\n"
        "\n"
        "==================== sample 1 source_row 1 ====================\n"
        "--- message 0 role=user ---\n"
        "INSTRUCTION\n"
        "\n"
        "PROMPT 1\n"
        "\n"
        "Let's think.\n"
    )


def test_write_formatted_prompt_sample_text_can_load_template_dir(tmp_path):
    neutral_parquet_path = tmp_path / "neutral.parquet"
    sample_output_path = tmp_path / "formatted_sample.txt"
    prompt_config_path = tmp_path / "prompt.yaml"
    template_dir = tmp_path / "templates" / "first_turn"
    template_dir.mkdir(parents=True)
    (template_dir / "a.jinja").write_text("A\n\n{{ problem }}\n", encoding="utf-8")
    (template_dir / "b.jinja").write_text("B\n\n{{ problem }}\n", encoding="utf-8")
    prompt_type = pa.list_(
        pa.struct(
            [
                pa.field("content", pa.string()),
                pa.field("role", pa.string()),
            ]
        )
    )
    neutral_prompts = [
        [{"role": "user", "content": "PROMPT 0"}],
        [{"role": "user", "content": "PROMPT 1"}],
    ]
    table = pa.table({"prompt": pa.array(neutral_prompts, type=prompt_type), "ability": ["kernel", "kernel"]})
    pq.write_table(table, neutral_parquet_path)
    prompt_config_path.write_text(
        "per_turn_prompts:\n"
        "  - name: first_turn\n"
        "    template_dir: templates/first_turn\n",
        encoding="utf-8",
    )

    sample_indices = write_formatted_prompt_sample_text(
        neutral_parquet_path,
        sample_output_path,
        prompt_config_path,
        sample_size=2,
        seed=0,
    )

    assert sample_indices == [0, 1]
    assert sample_output_path.read_text() == (
        "==================== sample 0 source_row 0 ====================\n"
        "--- message 0 role=user ---\n"
        "B\n"
        "\n"
        "PROMPT 0\n"
        "\n"
        "==================== sample 1 source_row 1 ====================\n"
        "--- message 0 role=user ---\n"
        "B\n"
        "\n"
        "PROMPT 1\n"
    )


def test_materialize_backend_neutral_parquet_writes_matching_text_copy(tmp_path):
    input_path = tmp_path / "source.parquet"
    output_path = tmp_path / "neutral.parquet"
    text_output_path = tmp_path / "neutral.txt"
    prompt_type = pa.list_(
        pa.struct(
            [
                pa.field("content", pa.string()),
                pa.field("role", pa.string()),
            ]
        )
    )
    source_prompts = [
        [
            {
                "role": "user",
                "content": """
You write custom Triton kernels.

You are given the following architecture:
```python
class Model:
    pass
```
""",
            }
        ]
    ]
    table = pa.table({"prompt": pa.array(source_prompts, type=prompt_type), "ability": ["kernel"]})
    pq.write_table(table, input_path)

    summary = materialize_backend_neutral_parquet(input_path, output_path, text_output_path)

    neutral_row = pq.read_table(output_path).to_pylist()[0]
    neutral_content = neutral_row["prompt"][0]["content"]
    text_content = text_output_path.read_text()
    assert summary.rows == 1
    assert summary.output_path == output_path
    assert summary.text_output_path == text_output_path
    assert neutral_content == "You are given the following PyTorch model:\n```python\nclass Model:\n    pass\n```"
    assert text_content == (
        "==================== row 0 ====================\n"
        "--- message 0 role=user ---\n"
        "You are given the following PyTorch model:\n"
        "```python\n"
        "class Model:\n"
        "    pass\n"
        "```\n"
    )
