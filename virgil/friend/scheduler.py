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
Scheduler system for Friend bot.

Handles recurring scheduled tasks like "always post X in Y channel" or
"always DM me this at X time of day".
"""

import asyncio
import os
import yaml
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional, Callable, List
import re


@dataclass
class ScheduledTask:
    """Represents a scheduled recurring task."""

    task_id: str
    task_type: str  # "post" or "dm"
    channel_id: Optional[int]
    channel_name: Optional[str]
    user_id: Optional[int]
    user_name: Optional[str]
    message: str
    schedule_type: str  # "daily", "hourly", "weekly", "interval"
    schedule_value: str  # Time string (e.g., "14:30") or interval (e.g., "1 hour")
    guild_id: Optional[int] = None
    guild_name: Optional[str] = None
    enabled: bool = True
    created_at: str = None
    last_executed: Optional[str] = None
    next_execution: Optional[str] = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.next_execution is None:
            self.next_execution = self._calculate_next_execution()

    def _calculate_next_execution(self) -> str:
        """Calculate the next execution time based on schedule."""
        now = datetime.now()

        if self.schedule_type == "daily":
            # Parse time string (e.g., "14:30")
            try:
                hour, minute = map(int, self.schedule_value.split(":"))
                scheduled_time = time(hour, minute)
                next_time = datetime.combine(now.date(), scheduled_time)

                # If time has passed today, schedule for tomorrow
                if next_time <= now:
                    next_time += timedelta(days=1)

                return next_time.isoformat()
            except Exception as e:
                print(f"Error parsing daily schedule time: {e}")
                return (now + timedelta(days=1)).isoformat()

        elif self.schedule_type == "hourly":
            # Schedule for next hour
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(
                hours=1
            )
            return next_hour.isoformat()

        elif self.schedule_type == "interval":
            # Parse interval (e.g., "1 hour", "30 mins")
            interval = self._parse_interval(self.schedule_value)
            if interval:
                return (now + interval).isoformat()
            else:
                return (now + timedelta(hours=1)).isoformat()

        elif self.schedule_type == "weekly":
            # Parse day of week and time (e.g., "monday 14:30")
            try:
                day_name, time_str = self.schedule_value.split(maxsplit=1)
                hour, minute = map(int, time_str.split(":"))
                scheduled_time = time(hour, minute)

                # Map day name to weekday (0=Monday, 6=Sunday)
                days = {
                    "monday": 0,
                    "tuesday": 1,
                    "wednesday": 2,
                    "thursday": 3,
                    "friday": 4,
                    "saturday": 5,
                    "sunday": 6,
                }
                target_day = days.get(day_name.lower(), 0)
                current_day = now.weekday()

                days_ahead = (target_day - current_day) % 7
                if days_ahead == 0:
                    # If today, check if time has passed
                    scheduled_datetime = datetime.combine(now.date(), scheduled_time)
                    if scheduled_datetime <= now:
                        days_ahead = 7  # Schedule for next week

                next_date = now.date() + timedelta(days=days_ahead)
                next_time = datetime.combine(next_date, scheduled_time)
                return next_time.isoformat()
            except Exception as e:
                print(f"Error parsing weekly schedule: {e}")
                return (now + timedelta(days=7)).isoformat()

        # Default: schedule for 1 hour from now
        return (now + timedelta(hours=1)).isoformat()

    def _parse_interval(self, interval_str: str) -> Optional[timedelta]:
        """Parse an interval string like '1 hour' or '30 mins'."""
        patterns = [
            (r"(\d+)\s*(?:minute|min|m)\s*(?:s)?", lambda x: timedelta(minutes=int(x))),
            (r"(\d+)\s*(?:hour|hr|h)\s*(?:s)?", lambda x: timedelta(hours=int(x))),
            (r"(\d+)\s*(?:day|d)\s*(?:s)?", lambda x: timedelta(days=int(x))),
            (r"(\d+)\s*(?:second|sec|s)\s*(?:s)?", lambda x: timedelta(seconds=int(x))),
        ]

        interval_lower = interval_str.lower()
        for pattern, converter in patterns:
            match = re.search(pattern, interval_lower)
            if match:
                value = int(match.group(1))
                return converter(value)

        return None

    def update_next_execution(self):
        """Update the next execution time after task execution."""
        self.last_executed = datetime.now().isoformat()
        self.next_execution = self._calculate_next_execution()

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        result = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "message": self.message,
            "schedule_type": self.schedule_type,
            "schedule_value": self.schedule_value,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_executed": self.last_executed,
            "next_execution": self.next_execution,
        }
        # Add guild info if present (optional for backward compatibility)
        if self.guild_id is not None:
            result["guild_id"] = self.guild_id
        if self.guild_name is not None:
            result["guild_name"] = self.guild_name
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create ScheduledTask from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            channel_id=data.get("channel_id"),
            channel_name=data.get("channel_name"),
            guild_id=data.get("guild_id"),
            guild_name=data.get("guild_name"),
            user_id=data.get("user_id"),
            user_name=data.get("user_name"),
            message=data["message"],
            schedule_type=data["schedule_type"],
            schedule_value=data["schedule_value"],
            enabled=data.get("enabled", True),
            created_at=data.get("created_at"),
            last_executed=data.get("last_executed"),
            next_execution=data.get("next_execution"),
        )


class Scheduler:
    """Manages scheduled recurring tasks."""

    def __init__(
        self, storage_file: str = "schedules.yaml", reload_interval: int = 300
    ):
        """
        Initialize the scheduler.

        Args:
            storage_file: Path to YAML file for persistent storage
            reload_interval: How often to reload schedules from file (seconds, default: 5 minutes)
        """
        self.storage_file = storage_file
        self.reload_interval = reload_interval
        self.tasks: dict[str, ScheduledTask] = {}
        self._next_id = 1
        self._execution_callback: Optional[Callable] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_reload = datetime.now()

        # Load existing schedules
        self._load_schedules()

    def set_execution_callback(self, callback: Callable):
        """Set the callback function to execute when a task is due."""
        self._execution_callback = callback

    def _load_schedules(self):
        """Load schedules from YAML file."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r") as f:
                    data = yaml.safe_load(f)
                    if data and "tasks" in data:
                        for task_data in data["tasks"]:
                            task = ScheduledTask.from_dict(task_data)
                            if task.enabled:
                                self.tasks[task.task_id] = task
                                # Update next_id to avoid conflicts
                                try:
                                    tid = int(task.task_id)
                                    if tid >= self._next_id:
                                        self._next_id = tid + 1
                                except ValueError:
                                    pass
            except Exception as e:
                print(f"Error loading schedules: {e}")
        else:
            # Create empty file
            self._save_schedules()

    def _save_schedules(self):
        """Save schedules to YAML file."""
        try:
            tasks_data = [task.to_dict() for task in self.tasks.values()]
            data = {"tasks": tasks_data}
            with open(self.storage_file, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error saving schedules: {e}")

    def add_task(
        self,
        task_type: str,
        message: str,
        schedule_type: str,
        schedule_value: str,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
        user_id: Optional[int] = None,
        user_name: Optional[str] = None,
        guild_id: Optional[int] = None,
        guild_name: Optional[str] = None,
    ) -> ScheduledTask:
        """
        Add a new scheduled task.

        Args:
            task_type: "post" or "dm"
            message: Message to send
            schedule_type: "daily", "hourly", "weekly", or "interval"
            schedule_value: Time string or interval (e.g., "14:30", "1 hour")
            channel_id: Channel ID for "post" tasks
            channel_name: Channel name for "post" tasks
            user_id: User ID for "dm" tasks
            user_name: User name for "dm" tasks
            guild_id: Discord guild/server ID (optional)
            guild_name: Discord guild/server name (optional)

        Returns:
            The created ScheduledTask object
        """
        task_id = str(self._next_id)
        self._next_id += 1

        task = ScheduledTask(
            task_id=task_id,
            task_type=task_type,
            channel_id=channel_id,
            channel_name=channel_name,
            guild_id=guild_id,
            guild_name=guild_name,
            user_id=user_id,
            user_name=user_name,
            message=message,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
        )

        self.tasks[task_id] = task
        self._save_schedules()

        return task

    def get_tasks_for_channel(self, channel_id: int) -> List[ScheduledTask]:
        """Get all scheduled tasks for a channel."""
        return [
            task
            for task in self.tasks.values()
            if task.enabled and task.channel_id == channel_id
        ]

    def get_tasks_for_user(self, user_id: int) -> List[ScheduledTask]:
        """Get all scheduled tasks for a user."""
        return [
            task
            for task in self.tasks.values()
            if task.enabled and task.user_id == user_id
        ]

    def get_all_tasks(self) -> List[ScheduledTask]:
        """Get all scheduled tasks."""
        return list(self.tasks.values())

    def remove_task(self, task_id: str):
        """Remove a scheduled task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save_schedules()

    def disable_task(self, task_id: str):
        """Disable a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            self._save_schedules()

    def enable_task(self, task_id: str):
        """Enable a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            self.tasks[task_id].next_execution = self.tasks[
                task_id
            ]._calculate_next_execution()
            self._save_schedules()

    def start(self):
        """Start the scheduler checking loop."""
        if not self._running:
            self._running = True
            # Get the current event loop
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self._check_schedules_loop())
            except RuntimeError:
                # No running loop - this will be called from async context
                # Try to get event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        self._task = loop.create_task(self._check_schedules_loop())
                    else:
                        # Store flag to start when loop is available
                        self._pending_start = True
                except RuntimeError:
                    self._pending_start = True

    def stop(self):
        """Stop the scheduler checking loop."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _check_schedules_loop(self):
        """Background task that checks for due scheduled tasks."""
        while self._running:
            try:
                # Reload schedules periodically
                if (
                    datetime.now() - self._last_reload
                ).total_seconds() >= self.reload_interval:
                    self._load_schedules()
                    self._last_reload = datetime.now()

                now = datetime.now()
                due_tasks = [
                    task
                    for task in self.tasks.values()
                    if task.enabled
                    and task.next_execution
                    and datetime.fromisoformat(task.next_execution) <= now
                ]

                for task in due_tasks:
                    await self._execute_task(task)

                # Sleep for a short time before checking again
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in schedule check loop: {e}")
                await asyncio.sleep(30)

    async def _execute_task(self, task: ScheduledTask):
        """Execute a due scheduled task."""
        if not task.enabled:
            return

        task.update_next_execution()
        self._save_schedules()

        if self._execution_callback:
            try:
                await self._execution_callback(task)
            except Exception as e:
                print(f"Error executing scheduled task callback: {e}")
                # Mark as not executed so it can be retried
                task.update_next_execution()
                self._save_schedules()


def parse_schedule_command(text: str) -> Optional[dict]:
    """
    Parse a schedule command like "always post X in Y channel" or "always DM me this at X time".

    Args:
        text: The schedule command text

    Returns:
        Dictionary with parsed schedule info or None if parsing fails
    """
    text_lower = text.lower().strip()

    # Pattern: "always post <message> in <channel> channel [at <time>|every <interval>]"
    # Also handle: "post <message> every <interval>" (assumes current channel)
    post_patterns = [
        r"always\s+post\s+(.+?)\s+in\s+(.+?)\s+channel\s+(?:at\s+)?(.+)",
        r"always\s+post\s+(.+?)\s+in\s+(.+?)\s+channel",
        r"schedule\s+post\s+(.+?)\s+in\s+(.+?)\s+channel\s+(?:at\s+)?(.+)",
        r"post\s+(.+?)\s+every\s+(.+)",  # "post X every Y" (uses current channel)
        r"can\s+you\s+post\s+(.+?)\s+every\s+(.+)",  # "can you post X every Y"
    ]

    # Pattern: "always DM me <message> at <time>" or "always DM me <message> every <interval>"
    dm_patterns = [
        r"always\s+dm\s+(?:me|us)\s+(.+?)\s+(?:at|every)\s+(.+)",
        r"schedule\s+dm\s+(?:me|us)\s+(.+?)\s+(?:at|every)\s+(.+)",
    ]

    # Try post patterns
    for i, pattern in enumerate(post_patterns):
        match = re.search(pattern, text_lower)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                message, channel, schedule = groups
                schedule = schedule.strip()
            elif len(groups) == 2:
                message, schedule_or_channel = groups
                # Patterns that end with "every" (indices 3, 4) always have schedule as second group
                # Patterns with "in ... channel" (indices 0, 1, 2) have channel as second group
                if i >= 3:  # Patterns like "post X every Y" or "can you post X every Y"
                    # Second group is always the schedule
                    schedule = schedule_or_channel.strip()
                    channel = None  # Will use current channel
                else:
                    # Check if second group looks like a schedule (contains time patterns, intervals, etc.)
                    if re.search(
                        r"\d+\s*(?:minute|min|m|hour|hr|h|day|d|second|sec|s)|at\s+\d|hourly|daily|weekly|every",
                        schedule_or_channel.lower(),
                    ):
                        # It's a schedule, channel is current channel (will be set later)
                        schedule = schedule_or_channel.strip()
                        channel = None  # Will use current channel
                    else:
                        # It's a channel name, use default schedule
                        channel = schedule_or_channel.strip()
                        schedule = "daily 09:00"  # Default to daily at 9 AM
            else:
                continue

            # Parse schedule
            schedule_info = _parse_schedule(schedule)
            if schedule_info:
                result = {
                    "task_type": "post",
                    "message": message.strip(),
                    "schedule_type": schedule_info["type"],
                    "schedule_value": schedule_info["value"],
                }
                if channel:
                    result["channel_name"] = channel.strip()
                else:
                    # Channel will be set from current context
                    result["channel_name"] = None
                return result

    # Try DM patterns
    for pattern in dm_patterns:
        match = re.search(pattern, text_lower)
        if match:
            message, schedule = match.groups()

            # Parse schedule
            schedule_info = _parse_schedule(schedule.strip())
            if schedule_info:
                return {
                    "task_type": "dm",
                    "message": message.strip(),
                    "schedule_type": schedule_info["type"],
                    "schedule_value": schedule_info["value"],
                }

    return None


def _parse_schedule(schedule_str: str) -> Optional[dict]:
    """Parse a schedule string into type and value."""
    schedule_lower = schedule_str.lower().strip()

    # Daily at specific time: "at 14:30" or "daily at 14:30" or "14:30"
    daily_match = re.search(r"(?:daily\s+)?(?:at\s+)?(\d{1,2}):(\d{2})", schedule_lower)
    if daily_match:
        hour, minute = daily_match.groups()
        return {"type": "daily", "value": f"{hour}:{minute}"}

    # Hourly: "hourly" or "every hour"
    if re.search(r"hourly|every\s+hour", schedule_lower):
        return {"type": "hourly", "value": ""}

    # Weekly: "monday at 14:30" or "every monday at 14:30"
    weekly_match = re.search(
        r"(?:every\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:at\s+)?(\d{1,2}):(\d{2})",
        schedule_lower,
    )
    if weekly_match:
        day, hour, minute = weekly_match.groups()
        return {"type": "weekly", "value": f"{day} {hour}:{minute}"}

    # Interval: "every 1 hour", "every 30 mins", "5 minutes", etc.
    # First try with "every" prefix
    interval_match = re.search(r"every\s+(.+)", schedule_lower)
    if interval_match:
        interval_str = interval_match.group(1)
        # Validate it's a valid interval
        if re.search(
            r"\d+\s*(?:minute|min|m|hour|hr|h|day|d|second|sec|s)", interval_str
        ):
            return {"type": "interval", "value": interval_str}

    # Also try without "every" prefix (e.g., "5 minutes", "30 seconds")
    # Match the full interval including number and unit
    interval_match_no_prefix = re.search(
        r"(\d+)\s*(minutes?|mins?|m|hours?|hrs?|h|days?|d|seconds?|secs?|s)",
        schedule_lower,
    )
    if interval_match_no_prefix:
        # Use the matched portion directly to preserve original format
        interval_str = interval_match_no_prefix.group(0)
        return {"type": "interval", "value": interval_str}

    return None
