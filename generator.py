from config import QA_PROMPT


class generator:
    def __init__(self, llm):
        self.llm = llm

    def answer(self, question, reference, suggestion=None, hint=None):
        suggestion_text = f"Here are some suggestions you need to follow: {suggestion}\n" if suggestion else ""
        hint_text = f"Here are some mistakes you may make, you need to be careful: {hint}\n" if hint else ""
        all_hint = suggestion_text + hint_text
        qa_prompt = QA_PROMPT.format(reference=reference, question=question, all_hint=all_hint)
        
        output = self.llm.get_output(qa_prompt, system_prompt="Act as an LLM helper.")

        try:
            new_output = eval(output)
            return str(new_output['reason']), str(new_output['answer'])
        except Exception as e:
            print(f"Error in generator!\n{output}\nException: {e}")
            return str(output), str(output)

        
