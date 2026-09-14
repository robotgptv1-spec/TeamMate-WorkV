import json
import os
import re
from dataclasses import dataclass, field


class QuotaExceededError(Exception):
    pass


def safe_json_parse(text):
    text = text.strip()

    # Remove markdown fences if provider returns them
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())

    return json.loads(text.strip())


class LLMProvider:

    def __init__(self, api_key):
        self.api_key = api_key

    def call(self, system, user):
        raise NotImplementedError

    def _check_quota_error(self, error):
        message = str(error).lower()

        quota_words = [
            "quota",
            "rate limit",
            "429",
            "insufficient_quota",
            "resource_exhausted",
            "too many requests"
        ]

        if any(word in message for word in quota_words):
            raise QuotaExceededError(str(error))


# ============================================================
# CLAUDE
# ============================================================

class ClaudeProvider(LLMProvider):

    def __init__(self, api_key):
        super().__init__(api_key)

        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=api_key
        )

    def call(self, system, user):

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8000,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": user
                    }
                ]
            )

            return response.content[0].text

        except Exception as e:
            self._check_quota_error(e)

            if "529" in str(e):
                raise QuotaExceededError(str(e))

            raise


# ============================================================
# OPENAI
# ============================================================

class OpenAIProvider(LLMProvider):

    def __init__(self, api_key):
        super().__init__(api_key)

        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key
        )

    def call(self, system, user):

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={
                    "type": "json_object"
                },
                messages=[
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    }
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            self._check_quota_error(e)
            raise


# ============================================================
# GEMINI
# ============================================================

class GeminiProvider(LLMProvider):

    def __init__(self, api_key):
        super().__init__(api_key)

        from google import genai

        self.client = genai.Client(
            api_key=api_key
        )

    def call(self, system, user):

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    system,
                    user
                ],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2
                }
            )

            return response.text

        except Exception as e:
            self._check_quota_error(e)
            raise


# ============================================================
# PROJECT STATE
# ============================================================

@dataclass
class ProjectState:

    prompt: str

    planning: str = ""

    files: list = field(
        default_factory=list
    )

    code: dict = field(
        default_factory=dict
    )

    review: dict = field(
        default_factory=dict
    )

    review_iteration: int = 0

    @property
    def pending_files(self):

        return [
            filename
            for filename in self.files
            if filename not in self.code
        ]

    def to_dict(self):

        return {
            "prompt": self.prompt,
            "planning": self.planning,
            "files": self.files,
            "code": self.code,
            "review": self.review,
            "review_iteration": self.review_iteration
        }

    def save(self, output_dir):

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        state_path = os.path.join(
            output_dir,
            "_state.json"
        )

        with open(
            state_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.to_dict(),
                f,
                indent=2,
                ensure_ascii=False
            )

    @classmethod
    def load(cls, output_dir):

        state_path = os.path.join(
            output_dir,
            "_state.json"
        )

        if not os.path.exists(state_path):
            return None

        with open(
            state_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return cls(
            prompt=data.get("prompt", ""),
            planning=data.get("planning", ""),
            files=data.get("files", []),
            code=data.get("code", {}),
            review=data.get("review", {}),
            review_iteration=data.get(
                "review_iteration",
                0
            )
        )


# ============================================================
# MULTI LLM WORK
# ============================================================

class MultiLLMWork:

    MAX_REVIEW_LOOPS = 3

    def __init__(
        self,
        prompt,
        output_dir,
        claude_key=None,
        openai_key=None,
        gemini_key=None,
        on_event=None
    ):

        self.prompt = prompt

        self.output_dir = output_dir

        self.on_event = on_event

        self.providers = []

        # ----------------------------------------------------
        # Provider Chain
        # ----------------------------------------------------

        if claude_key:
            try:

                self.providers.append(
                    (
                        "Claude",
                        ClaudeProvider(claude_key)
                    )
                )

            except Exception as e:

                self._log(
                    f"⚠️ Claude initialization failed: {e}"
                )

        if openai_key:
            try:

                self.providers.append(
                    (
                        "OpenAI",
                        OpenAIProvider(openai_key)
                    )
                )

            except Exception as e:

                self._log(
                    f"⚠️ OpenAI initialization failed: {e}"
                )

        if gemini_key:
            try:

                self.providers.append(
                    (
                        "Gemini",
                        GeminiProvider(gemini_key)
                    )
                )

            except Exception as e:

                self._log(
                    f"⚠️ Gemini initialization failed: {e}"
                )

        # ----------------------------------------------------
        # Resume Existing Project
        # ----------------------------------------------------

        existing_state = ProjectState.load(
            self.output_dir
        )

        if existing_state:

            self.state = existing_state

            self._log(
                "🔄 Existing project state loaded."
            )

        else:

            self.state = ProjectState(
                prompt=prompt
            )

    # ========================================================
    # LOGGING
    # ========================================================

    def _log(self, message):

        print(message)

        if self.on_event:

            self.on_event(
                {
                    "type": "log",
                    "message": message
                }
            )

    # ========================================================
    # RUN STEP
    # ========================================================

    def _run_step(self, system, user):

        if not self.providers:

            raise RuntimeError(
                "❌ Koi LLM provider configured nahi hai."
            )

        for provider_name, provider in self.providers:

            try:

                self._log(
                    f"🤖 Using provider: {provider_name}"
                )

                raw_response = provider.call(
                    system,
                    user
                )

                result = safe_json_parse(
                    raw_response
                )

                self._log(
                    f"✅ {provider_name} response received."
                )

                return result

            except QuotaExceededError:

                self._log(
                    f"⚠️ {provider_name} quota/rate limit exceeded. "
                    f"Failing over..."
                )

                continue

            except json.JSONDecodeError:

                self._log(
                    f"⚠️ {provider_name} returned invalid JSON. "
                    f"Trying next provider..."
                )

                continue

            except Exception as e:

                self._log(
                    f"❌ {provider_name} failed: {e}"
                )

                continue

        raise RuntimeError(
            "❌ All configured LLM providers failed."
        )

    # ========================================================
    # PLANNER
    # ========================================================

    def _plan(self):

        self._log(
            "🧠 Planner: Creating project architecture..."
        )

        system = """
You are the Lead Architect.

Analyze the user's project requirement.

Create a clean, modular and production-oriented
software architecture.

Return ONLY valid JSON.

Do not write actual code.

The JSON format must be:

{
    "planning": "Detailed architecture and implementation plan",
    "files": [
        "filename1.py",
        "filename2.py",
        "index.html"
    ]
}
"""

        user = f"""
User Project Requirement:

{self.state.prompt}

Design the complete project architecture.
"""

        result = self._run_step(
            system,
            user
        )

        self.state.planning = result.get(
            "planning",
            ""
        )

        self.state.files = result.get(
            "files",
            []
        )

        self.state.save(
            self.output_dir
        )

        self._log(
            f"📋 Plan created with {len(self.state.files)} files."
        )

    # ========================================================
    # CODER
    # ========================================================

    def _code(self):

        pending = self.state.pending_files

        if not pending:

            self._log(
                "✅ No pending files."
            )

            return

        self._log(
            f"👨‍💻 Coder: Working on {len(pending)} files..."
        )

        system = """
You are the Lead Programmer.

Build the project according to the architecture.

Write production-ready code.

Follow:

- clean architecture
- modularity
- SOLID principles
- security best practices
- error handling
- maintainability
- scalability

Return ONLY valid JSON.

Format:

{
    "code": {
        "filename.ext": "complete file content"
    }
}

Do not use markdown fences.
"""

        user = f"""
PROJECT REQUIREMENT:

{self.state.prompt}


PROJECT PLAN:

{self.state.planning}


ALL PROJECT FILES:

{json.dumps(self.state.files, indent=2)}


ALREADY COMPLETED FILES:

{json.dumps(self.state.code, indent=2)}


PENDING FILES:

{json.dumps(pending, indent=2)}


REVIEWER FEEDBACK:

{json.dumps(self.state.review, indent=2)}


IMPORTANT:

If reviewer feedback exists, fix those issues.

Do not blindly rewrite working code.

Only return files that need to be created or modified.
"""

        result = self._run_step(
            system,
            user
        )

        new_code = result.get(
            "code",
            {}
        )

        if not new_code:

            self._log(
                "⚠️ Provider ne koi code return nahi kiya."
            )

            return

        # Merge new/updated files
        self.state.code.update(
            new_code
        )

        self.state.save(
            self.output_dir
        )

        self._write_files(
            new_code
        )

        self._log(
            f"✅ Coder completed {len(new_code)} files."
        )

    # ========================================================
    # REVIEWER
    # ========================================================

    def _review_code(self):

        if not self.state.code:

            self._log(
                "⚠️ Koi code nahi hai review ke liye."
            )

            return False

        self._log(
            f"🔍 Reviewer: Reviewing code "
            f"(iteration {self.state.review_iteration})..."
        )

        system = """
You are the Senior Code Reviewer.

Review the complete project.

Check:

- correctness
- bugs
- architecture
- security
- maintainability
- scalability
- error handling
- code quality
- integration issues

Return ONLY valid JSON.

Required format:

{
    "approved": true,
    "review": {
        "filename.py": "specific review comments"
    }
}

Rules:

approved = true
ONLY when the project is ready.

approved = false
when changes are required.

If approved is false,
provide specific actionable feedback.

Do not use markdown fences.
"""

        user = f"""
PROJECT REQUIREMENT:

{self.state.prompt}


PROJECT PLAN:

{self.state.planning}


PROJECT CODE:

{json.dumps(self.state.code, indent=2)}


Review the complete project.
"""

        result = self._run_step(
            system,
            user
        )

        approved = result.get(
            "approved",
            False
        )

        new_review = result.get(
            "review",
            {}
        )

        # ----------------------------------------------------
        # Save Review Feedback
        # ----------------------------------------------------

        self.state.review = new_review

        self.state.save(
            self.output_dir
        )

        if approved:

            self._log(
                "✅ Reviewer: Project APPROVED."
            )

        else:

            self._log(
                "❌ Reviewer: Changes required."
            )

            if new_review:

                for filename, feedback in new_review.items():

                    self._log(
                        f"📝 {filename}: {feedback}"
                    )

        return approved

    # ========================================================
    # WRITE FILES
    # ========================================================

    def _write_files(self, files):

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        for filename, content in files.items():

            file_path = os.path.join(
                self.output_dir,
                filename
            )

            # Prevent accidental directory traversal
            safe_path = os.path.abspath(
                file_path
            )

            output_root = os.path.abspath(
                self.output_dir
            )

            if not safe_path.startswith(
                output_root + os.sep
            ):

                self._log(
                    f"⚠️ Unsafe file path skipped: {filename}"
                )

                continue

            os.makedirs(
                os.path.dirname(safe_path),
                exist_ok=True
            )

            with open(
                safe_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(content)

            self._log(
                f"📄 File written: {filename}"
            )

            if self.on_event:

                self.on_event(
                    {
                        "type": "file_written",
                        "filename": filename
                    }
                )

    # ========================================================
    # BUILD
    # ========================================================

    def build(self):

        self._log(
            f"🚀 MultiLLMWork started with "
            f"{len(self.providers)} providers."
        )

        # ----------------------------------------------------
        # STEP 1 — PLAN
        # ----------------------------------------------------

        if not self.state.planning:

            self._plan()

        else:

            self._log(
                "⏭️ Existing plan found. Skipping planner."
            )

        # ----------------------------------------------------
        # REVIEW / CODING LOOP
        # ----------------------------------------------------

        for iteration in range(
            self.MAX_REVIEW_LOOPS
        ):

            self.state.review_iteration = iteration + 1

            self._log(
                f"\n🔁 Agent Loop "
                f"{iteration + 1}/{self.MAX_REVIEW_LOOPS}"
            )

            # ------------------------------------------------
            # CODE
            # ------------------------------------------------

            self._code()

            # ------------------------------------------------
            # REVIEW
            # ------------------------------------------------

            approved = self._review_code()

            # ------------------------------------------------
            # APPROVED
            # ------------------------------------------------

            if approved:

                self._log(
                    "🎉 Project passed code review."
                )

                state_path = os.path.join(
                    self.output_dir,
                    "_state.json"
                )

                if os.path.exists(state_path):

                    os.remove(
                        state_path
                    )

                if self.on_event:

                    self.on_event(
                        {
                            "type": "done",
                            "message": "Project completed successfully."
                        }
                    )

                return

            # ------------------------------------------------
            # REJECTED
            # ------------------------------------------------

            self._log(
                "🔄 Reviewer feedback will be sent "
                "back to the coder."
            )

            # Clear only after coder has consumed it.
            # We intentionally keep it here so the next
            # iteration receives the feedback.

        # ----------------------------------------------------
        # MAX ITERATIONS REACHED
        # ----------------------------------------------------

        self._log(
            f"⚠️ Maximum review loops "
            f"({self.MAX_REVIEW_LOOPS}) reached."
        )

        self._log(
            "⏸️ Project paused for manual review."
        )

        self.state.save(
            self.output_dir
        )

        if self.on_event:

            self.on_event(
                {
                    "type": "paused",
                    "message": (
                        "Maximum review iterations reached."
                    )
                }
            )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    prompt = """
Create a simple digital clock web application.

Requirements:

- HTML
- CSS
- JavaScript
- Real-time clock
- Clean UI
"""

    output_dir = "test_project"

    team = MultiLLMWork(
        prompt=prompt,
        output_dir=output_dir,
        claude_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_key=os.getenv("OPENAI_API_KEY"),
        gemini_key=os.getenv("GEMINI_API_KEY")
    )

    team.build()