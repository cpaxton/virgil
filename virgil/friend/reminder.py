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
            self._task = asyncio.create_task(self._check_reminders_loop())

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
    # Patterns for time parsing
    patterns = [
        (r"(\d+)\s*(?:minute|min|m)\s*(?:s)?", lambda x: timedelta(minutes=int(x))),
        (r"(\d+)\s*(?:hour|hr|h)\s*(?:s)?", lambda x: timedelta(hours=int(x))),
        (r"(\d+)\s*(?:day|d)\s*(?:s)?", lambda x: timedelta(days=int(x))),
        (r"(\d+)\s*(?:second|sec|s)\s*(?:s)?", lambda x: timedelta(seconds=int(x))),
    ]

    text_lower = text.lower()

    for pattern, converter in patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = int(match.group(1))
            return converter(value)

    return None


def parse_reminder_command(text: str) -> Optional[tuple[timedelta, str]]:
    """
    Parse a reminder command like "remind me in 30 mins to do the dishes".

    Args:
        text: The reminder command text

    Returns:
        Tuple of (timedelta, message) or None if parsing fails
    """
    text_lower = text.lower().strip()

    # Pattern: "remind me in X to Y" or "remind me in X that Y"
    patterns = [
        r"remind\s+(?:me|us)?\s+in\s+(.+?)\s+(?:to|that)\s+(.+)",
        r"remind\s+(?:me|us)?\s+(.+?)\s+(?:to|that)\s+(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            time_str = match.group(1)
            message = match.group(2).strip()

            # Try to parse the time
            time_delta = parse_reminder_time(time_str)
            if time_delta:
                return (time_delta, message)

    return None
