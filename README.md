# 🎲 MetaNIMArena

A lightweight playground for **LLM‐powered impartial games**.  
The repo currently ships five classic combinatorial arenas:

| Game | Variant / Goal | Notice |
|------|----------------|--------|
| **NIM** | normal & misère | - |
| **Fibonacci** | normal & misère | - |
| **Kayles** | single & two-row pins | Under construction |
| **Chomp** | 2-D chocolate grid | - |
| **Corner Queen** | reach lower-left corner | - |

---

## ⚠️ Quick notes

* **Opponent model** By default the simulator pits your agent against `gpt-4o`.  
  That means **OpenAI API calls (and cost) are incurred** ⇒ set `OPENAI_API_KEY` in your environment.  
  Prefer zero-cost? Swap in any local model (Llama 3, Gemma, etc.) with the `--opponent_model` flag.
* **Gemini quirk** Some Gemini versions return streaming chunks that break the parser; GPT-series work out-of-the-box.
* **Under Construction** We are fixing the error and integrating the methods. Specifically, Kayles makes lots of error when using gemini models. We are now fixing it. For further, we will integrate all methods for easy usage.
* **Our results** are conducted between 2024.12. ~ 2025.01. And the results for Standard prompting and MAD2 is conducted during 2025.04. using gpt and gemini series.

## Environment
Install the required python library by running:
```bash
bash install.sh
```

## Inference

Here is an example command for running our NIM-Normal environments:
* This code requires your OpenAI api key or GEMINI api key, which means that cost will be charged.
```bash
python nim_normal.py --agent1_model=${MODEL_NAME} --agent1_prompt=${METHOD_NAME} --num_games=${NUM_GAMES} --temperature=${TEMPERATURE} --max_take=${MAX_TAKE}
```

To run with our `DREAMAD` method using `GPT-4o-mini`, use the following command:
```bash
python nim_normal.py --agent1_model=gpt-4o-mini --agent1_prompt=dreamad
```

You can also run other baselines by changing the `--agent1_prompt` argument as follows:
```bash
python nim_normal.py --agent1_model=gpt-4o-mini --agent1_prompt=self_consistency
```

## 📄 Citation

If you find this work useful in your research, please consider citing our paper:

### BibTeX
```bibtex
@misc{oh2025understandingbiasreinforcementllm,
      title={Understanding Bias Reinforcement in LLM Agents Debate}, 
      author={Jihwan Oh and Minchan Jeong and Jongwoo Ko and Se-Young Yun},
      year={2025},
      eprint={2503.16814},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.16814}, 
}
