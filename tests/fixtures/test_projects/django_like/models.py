"""Django-style models for testing."""

from typing import Optional, List
from .base import Model, Field


class User(Model):
    """User model."""
    
    username = Field(max_length=100, unique=True)
    email = Field(max_length=255, unique=True)
    is_active = Field(default=True)
    
    class Meta:
        db_table = "users"
        ordering = ["username"]
    
    def get_full_name(self) -> str:
        """Get user's full name."""
        return self.username.title()
    
    def deactivate(self) -> None:
        """Deactivate the user."""
        self.is_active = False
        self.save()


class Post(Model):
    """Blog post model."""
    
    title = Field(max_length=200)
    content = Field()
    author = Field(foreign_key=User)
    created_at = Field(auto_now_add=True)
    
    class Meta:
        db_table = "posts"
        ordering = ["-created_at"]
    
    def get_absolute_url(self) -> str:
        """Get post URL."""
        return f"/posts/{self.id}/"
    
    @property
    def summary(self) -> str:
        """Get post summary."""
        return self.content[:100] + "..."


class Comment(Model):
    """Comment on a post."""
    
    post = Field(foreign_key=Post)
    author = Field(foreign_key=User)
    content = Field()
    created_at = Field(auto_now_add=True)
    
    class Meta:
        db_table = "comments"
        ordering = ["created_at"]
    
    def notify_author(self) -> None:
        """Notify post author about new comment."""
        from .tasks import send_notification
        send_notification(self.post.author, f"New comment on {self.post.title}")