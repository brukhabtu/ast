"""Django-style views."""

from typing import Dict, Any
from .models import User, Post, Comment
from .utils import render_template, get_object_or_404


def index(request) -> Dict[str, Any]:
    """Homepage view."""
    posts = Post.objects().filter(is_published=True).all()
    return render_template("index.html", {"posts": posts})


def post_detail(request, post_id: int) -> Dict[str, Any]:
    """Post detail view."""
    post = get_object_or_404(Post, id=post_id)
    comments = Comment.objects().filter(post=post).all()
    return render_template("post_detail.html", {
        "post": post,
        "comments": comments
    })


class UserView:
    """User profile view."""
    
    def get(self, request, username: str) -> Dict[str, Any]:
        """Get user profile."""
        user = get_object_or_404(User, username=username)
        posts = Post.objects().filter(author=user).all()
        return render_template("user_profile.html", {
            "user": user,
            "posts": posts
        })
    
    def post(self, request, username: str) -> Dict[str, Any]:
        """Update user profile."""
        user = get_object_or_404(User, username=username)
        # Update logic here
        return self.get(request, username)


class PostCreateView:
    """Create new post view."""
    
    model = Post
    template_name = "post_form.html"
    
    def get(self, request) -> Dict[str, Any]:
        """Show post creation form."""
        return render_template(self.template_name, {"form": {}})
    
    def post(self, request) -> Dict[str, Any]:
        """Handle post creation."""
        data = request.POST
        post = self.model(
            title=data["title"],
            content=data["content"],
            author=request.user
        )
        post.save()
        return {"redirect": post.get_absolute_url()}