# Copyright 2024 Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# (c) 2024 by Chris Paxton

"""
Reminder system for Friend bot.

Handles scheduling and executing reminders with context about how the user
contacted the bot (channel, user info, etc.).
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Callable
import re


@dataclass
class Reminder:
    """Represents a reminder with all context."""

    reminder_id: str
    channel_id: Optional[int]
    channel_name: Optional[str]
    user_id: int
    user_name: str
    reminder_time: datetime
    message: str
    created_at: datetime
    executed: bool = False

    def to_dict(self):
        """Convert reminder to dictionary for serialization."""
        return {
            "reminder_id": self.reminder_id,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "reminder_time": self.reminder_time.isoformat(),
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "executed": self.executed,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create reminder from dictionary."""
        return cls(
            reminder_id=data["reminder_id"],
            channel_id=data.get("channel_id"),
            channel_name=data.get("channel_name"),
            user_id=data["user_id"],
            user_name=data["user_name"],
            reminder_time=datetime.fromisoformat(data["reminder_time"]),
            message=data["message"],
            created_at=datetime.fromisoformat(data["created_at"]),
            executed=data.get("executed", False),
        )


class ReminderManager:
    """Manages reminders: scheduling, storage, and execution."""

    def __init__(self, storage_file: str = "reminders.json"):
        """
        Initialize the reminder manager.

        Args:
            storage_file: Path to file for persistent storage of reminders
        """
        self.storage_file = storage_file
        self.reminders: dict[str, Reminder] = {}
        self._next_id = 1
        self._execution_callback: Optional[Callable] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Load existing reminders
        self._load_reminders()

    def set_execution_callback(self, callback: Callable):
        """Set the callback function to execute when a reminder is due."""
        self._execution_callback = callback

    def _load_reminders(self):
        """Load reminders from storage file."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r") as f:
                    data = json.load(f)
                    for reminder_data in data:
                        reminder = Reminder.from_dict(reminder_data)
                        if (
                            not reminder.executed
                            and reminder.reminder_time > datetime.now()
                        ):
                            self.reminders[reminder.reminder_id] = reminder
                            # Update next_id to avoid conflicts
                            try:
                                rid = int(reminder.reminder_id)
                                if rid >= self._next_id:
                                    self._next_id = rid + 1
                            except ValueError:
                                pass
            except Exception as e:
                print(f"Error loading reminders: {e}")

    def _save_reminders(self):
        """Save reminders to storage file."""
        try:
            reminders_data = [r.to_dict() for r in self.reminders.values()]
            with open(self.storage_file, "w") as f:
                json.dump(reminders_data, f, indent=2)
        except Exception as e:
            print(f"Error saving reminders: {e}")

    def add_reminder(
        self,
        channel_id: Optional[int],
        channel_name: Optional[str],
        user_id: int,
        user_name: str,
        reminder_time: datetime,
        message: str,
    ) -> Reminder:
        """
        Add a new reminder.

        Args:
            channel_id: Discord channel ID (None for DMs)
            channel_name: Discord channel name
            user_id: Discord user ID
            user_name: Discord user name/display name
            reminder_time: When to execute the reminder
            message: The reminder message to send

        Returns:
            The created Reminder object
        """
        reminder_id = str(self._next_id)
        self._next_id += 1

        reminder = Reminder(
            reminder_id=reminder_id,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            reminder_time=reminder_time,
            message=message,
            created_at=datetime.now(),
        )

        self.reminders[reminder_id] = reminder
        self._save_reminders()

        return reminder

    def get_reminders_for_user(self, user_id: int) -> list[Reminder]:
        """Get all active reminders for a user."""
        return [
            r
            for r in self.reminders.values()
            if r.user_id == user_id and not r.executed
        ]

    def remove_reminder(self, reminder_id: str):
        """Remove a reminder."""
        if reminder_id in self.reminders:
            del self.reminders[reminder_id]
            self._save_reminders()

    def start(self):
        """Start the reminder checking loop."""
        if not self._running:
            self._running = True
            # Get the current event loop
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self._check_reminders_loop())
            except RuntimeError:
                # No running loop - this will be called from async context
                # Try to get event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        self._task = loop.create_task(self._check_reminders_loop())
                    else:
                        # Store flag to start when loop is available
                        self._pending_start = True
                except RuntimeError:
                    self._pending_start = True

    def stop(self):
        """Stop the reminder checking loop."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _check_reminders_loop(self):
        """Background task that checks for due reminders."""
        while self._running:
            try:
                now = datetime.now()
                due_reminders = [
                    r
                    for r in self.reminders.values()
                    if not r.executed and r.reminder_time <= now
                ]

                for reminder in due_reminders:
                    await self._execute_reminder(reminder)

                # Sleep for a short time before checking again
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in reminder check loop: {e}")
                await asyncio.sleep(10)

    async def _execute_reminder(self, reminder: Reminder):
        """Execute a due reminder."""
        if reminder.executed:
            return

        reminder.executed = True
        self._save_reminders()

        if self._execution_callback:
            try:
                await self._execution_callback(reminder)
            except Exception as e:
                print(f"Error executing reminder callback: {e}")
                # Mark as not executed so it can be retried
                reminder.executed = False
                self._save_reminders()


def parse_reminder_time(text: str) -> Optional[timedelta]:
    """
    Parse a time duration from text like "30 mins", "1 hour", "2 hours", etc.

    Args:
        text: Text containing time duration

    Returns:
        timedelta object or None if parsing fails
    """
    # Patterns for time parsing (more flexible, including common typos)
    patterns = [
        (
            r"(\d+)\s*(?:minute|min|m|minuts|minute)\s*(?:s)?",
            lambda x: timedelta(minutes=int(x)),
        ),
        (
            r"(\d+)\s*(?:hour|hr|h|houres|hour)\s*(?:s)?",
            lambda x: timedelta(hours=int(x)),
        ),
        (r"(\d+)\s*(?:day|d)\s*(?:s)?", lambda x: timedelta(days=int(x))),
        (
            r"(\d+)\s*(?:second|sec|s|secnds|seconds)\s*(?:s)?",
            lambda x: timedelta(seconds=int(x)),
        ),
        (r"(\d+)\s*(?:week|wk|w)\s*(?:s)?", lambda x: timedelta(weeks=int(x))),
    ]

    text_lower = text.lower().strip()

    for pattern, converter in patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = int(match.group(1))
            return converter(value)

    return None


def parse_absolute_time(text: str) -> Optional[datetime]:
    """
    Parse an absolute time from text like "at 3pm", "at 14:30", "tomorrow at 9am", etc.

    Args:
        text: Text containing absolute time

    Returns:
        datetime object or None if parsing fails
    """

    text_lower = text.lower().strip()
    now = datetime.now()

    # Pattern: "at HH:MM" or "at H:MM" (24-hour or 12-hour)
    time_pattern = r"at\s+(\d{1,2}):(\d{2})\s*(?:am|pm)?"
    match = re.search(time_pattern, text_lower)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))

        # Check for AM/PM
        if "pm" in text_lower and hour < 12:
            hour += 12
        elif "am" in text_lower and hour == 12:
            hour = 0

        # Check if time has passed today, if so schedule for tomorrow
        target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target_time <= now:
            target_time += timedelta(days=1)

        return target_time

    # Pattern: "at Xpm" or "at Xam" (simple hour)
    simple_time_pattern = r"at\s+(\d{1,2})\s*(am|pm)"
    match = re.search(simple_time_pattern, text_lower)
    if match:
        hour = int(match.group(1))
        am_pm = match.group(2)

        if am_pm == "pm" and hour < 12:
            hour += 12
        elif am_pm == "am" and hour == 12:
            hour = 0

        target_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if target_time <= now:
            target_time += timedelta(days=1)

        return target_time

    # Pattern: "tomorrow at X"
    if "tomorrow" in text_lower:
        tomorrow = now + timedelta(days=1)
        # Try to extract time
        time_match = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", text_lower)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            am_pm = time_match.group(3)

            if am_pm == "pm" and hour < 12:
                hour += 12
            elif am_pm == "am" and hour == 12:
                hour = 0

            return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return None


def parse_reminder_command_with_llm(
    text: str, llm_prompt_func=None
) -> tuple[Optional[dict], Optional[str]]:
    """
    Use LLM to parse a reminder command when regex parsing fails.

    Args:
        text: The reminder command text
        llm_prompt_func: Optional function to call LLM with a prompt

    Returns:
        Tuple of (parsed_dict, parsing_instructions) or (None, None)
        parsing_instructions contains suggestions for improving parsing
    """
    if not llm_prompt_func:
        return None, None

    prompt = f"""Parse this reminder request and extract the time and message. Return ONLY a JSON object with this exact structure:
{{
    "time_delta": "HH:MM:SS" or null (for relative times like "in 30 mins"),
    "reminder_time": "YYYY-MM-DD HH:MM:SS" or null (for absolute times like "at 3pm"),
    "message": "the reminder message text",
    "users": ["list", "of", "user", "names"] or [] (empty if just "me"),
    "parsing_instructions": "specific regex patterns or parsing rules that would help parse similar requests in the future"
}}

Rules:
- If time is relative (e.g., "in 30 mins", "in 2 hours", "in 30 secnds"), set time_delta as "HH:MM:SS" format and reminder_time as null
  * Handle common typos: "secnds" → "seconds", "minuts" → "minutes", "houres" → "hours"
- If time is absolute (e.g., "at 3pm", "tomorrow at 9am"), set reminder_time as "YYYY-MM-DD HH:MM:SS" and time_delta as null
- Extract only the reminder message, not the time
- If the message appears incomplete (e.g., ends with "to" without a following phrase), set message to null
- If multiple users mentioned, list them in users array; if just "me" or "us", use empty array
- In parsing_instructions, suggest specific regex patterns or parsing improvements
- Return ONLY the JSON, no other text, no explanations, no markdown formatting

User request: "{text}"
"""

    try:
        response = llm_prompt_func(prompt)
        # Extract JSON from response (might be wrapped in markdown code blocks)
        import json
        import re

        # Try to find JSON in the response - look for complete JSON objects
        # First try to find JSON between ```json and ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object with balanced braces
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(0)
            else:
                # Last resort: try the whole response
                json_str = response.strip()

        parsed = json.loads(json_str)

        # Validate that we got meaningful data
        if not parsed.get("time_delta") and not parsed.get("reminder_time"):
            # No time parsed, return None
            return None, None

        # Check if message is incomplete or invalid
        message = parsed.get("message", "").strip()
        if (
            not message
            or len(message) < 2
            or message.lower() in ["to", "that", "about"]
        ):
            # Message is incomplete or just a connector word
            return None, None

        # Extract parsing instructions
        parsing_instructions = parsed.pop("parsing_instructions", None)

        # Convert time_delta string to timedelta if present
        if parsed.get("time_delta"):
            time_parts = parsed["time_delta"].split(":")
            if len(time_parts) == 3:
                hours, minutes, seconds = map(int, time_parts)
                parsed["time_delta"] = timedelta(
                    hours=hours, minutes=minutes, seconds=seconds
                )
            else:
                parsed["time_delta"] = None

        # Convert reminder_time string to datetime if present
        if parsed.get("reminder_time"):
            try:
                parsed["reminder_time"] = datetime.fromisoformat(
                    parsed["reminder_time"].replace(" ", "T")
                )
            except (ValueError, TypeError):
                parsed["reminder_time"] = None

        return parsed, parsing_instructions
    except Exception as e:
        print(f"Error parsing reminder with LLM: {e}")
        return None, None


def parse_reminder_command(text: str) -> Optional[dict]:
    """
    Parse a reminder command like "remind me in 30 mins to do the dishes"
    or "remind Chris and Julian at 3pm to do the dishes".

    Args:
        text: The reminder command text

    Returns:
        Dictionary with keys:
        - 'time_delta': timedelta (if relative time) or None
        - 'reminder_time': datetime (if absolute time) or None
        - 'message': str - the reminder message
        - 'users': list[str] - list of user names mentioned (empty if just "me")
        or None if parsing fails
    """
    text_lower = text.lower().strip()

    # Extract user mentions (names or "me"/"us")
    # Pattern: "remind [user1] [and user2] ... [in/at] ..."
    # Use non-greedy match to stop at "in" or "at"
    user_pattern = r"remind\s+(.+?)\s+(?:in|at|to|that)"
    user_match = re.search(user_pattern, text_lower)
    users = []
    if user_match:
        users_str = user_match.group(1).strip()
        # Split by "and", "&", comma
        users_list = re.split(r"\s+(?:and|&|,)\s+", users_str)
        users = [u.strip() for u in users_list if u.strip()]
        # Normalize "me" and "us"
        if "me" in users or "us" in users:
            users = []  # Empty means current user
    else:
        # Default to "me" if no users specified
        users = []

    # Try absolute time first: "remind [users] at X to Y" or "remind [users] at X: Y"
    absolute_patterns = [
        r"remind\s+(?:.+?)?\s+at\s+(.+?)\s+(?:to|that|:)\s+(.+)",
        r"remind\s+(?:.+?)?\s+tomorrow\s+(?:at\s+)?(.+?)\s+(?:to|that|:)\s+(.+)",
    ]

    for pattern in absolute_patterns:
        match = re.search(pattern, text_lower)
        if match:
            time_str = match.group(1)
            message = match.group(2).strip()

            # Try to parse absolute time
            reminder_time = parse_absolute_time(f"at {time_str}")
            if reminder_time:
                return {
                    "time_delta": None,
                    "reminder_time": reminder_time,
                    "message": message,
                    "users": users,
                }

    # Try relative time: "remind [users] in X to Y" or "remind [users] in X: Y"
    relative_patterns = [
        r"remind\s+(?:.+?)?\s+in\s+(.+?)\s*[:]\s*(.+)",  # Colon separator
        r"remind\s+(?:.+?)?\s+in\s+(.+?)\s+(?:to|that)\s+(.+)",  # "to" or "that"
        r"remind\s+(?:.+?)?\s+(.+?)\s*[:]\s*(.+)",  # Fallback with colon
        r"remind\s+(?:.+?)?\s+(.+?)\s+(?:to|that)\s+(.+)",  # Fallback: "remind me X to Y"
    ]

    for pattern in relative_patterns:
        match = re.search(pattern, text_lower)
        if match:
            time_str = match.group(1)
            message = match.group(2).strip()

            # Try to parse relative time
            time_delta = parse_reminder_time(time_str)
            if time_delta:
                return {
                    "time_delta": time_delta,
                    "reminder_time": None,
                    "message": message,
                    "users": users,
                }

    # Fallback: try to extract time and message from simpler patterns
    # "in 30 mins to do X" or "at 3pm to do X" or "in 30 mins: do X"
    simple_patterns = [
        r"in\s+(.+?)\s+(?:to|that|about|:)\s+(.+)",
        r"at\s+(.+?)\s+(?:to|that|about|:)\s+(.+)",
    ]

    for pattern in simple_patterns:
        match = re.search(pattern, text_lower)
        if match:
            time_str = match.group(1)
            message = match.group(2).strip()

            # Try absolute time first
            reminder_time = parse_absolute_time(f"at {time_str}")
            if reminder_time:
                return {
                    "time_delta": None,
                    "reminder_time": reminder_time,
                    "message": message,
                    "users": users,
                }

            # Try relative time
            time_delta = parse_reminder_time(time_str)
            if time_delta:
                return {
                    "time_delta": time_delta,
                    "reminder_time": None,
                    "message": message,
                    "users": users,
                }

    return None
