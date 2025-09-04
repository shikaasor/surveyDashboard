"""
Authentication utilities for admin access.

This module provides simple password-based authentication for admin features.
"""

import streamlit as st
import hashlib
from typing import Optional


def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(expected_password: Optional[str]) -> bool:
    """
    Check password using Streamlit's built-in password input.
    
    Args:
        expected_password: Expected password value
        
    Returns:
        True if password is correct, False otherwise
    """
    if not expected_password:
        st.warning("No admin password configured. Using default: 'admin'")
        expected_password = "admin"
    
    def password_entered():
        """Callback for password input."""
        if st.session_state["password"] == expected_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("Password incorrect")
        return False
    else:
        # Password correct
        return True


def require_admin_access(expected_password: Optional[str] = None):
    """
    Decorator to require admin access for a function.
    
    Args:
        expected_password: Expected password value
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if check_password(expected_password):
                return func(*args, **kwargs)
            else:
                st.stop()
        return wrapper
    return decorator


def logout():
    """Log out the current admin user."""
    if "password_correct" in st.session_state:
        del st.session_state["password_correct"]
    if "is_admin" in st.session_state:
        st.session_state["is_admin"] = False
    st.success("Logged out successfully")
    st.rerun()