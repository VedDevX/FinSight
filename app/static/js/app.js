// app/static/js/app.js

// Show loader overlay (used on form submit)
function showLoader() {
  const loader = document.getElementById("loader");
  if (loader) loader.classList.remove("hidden");
}

// Animate result reveal (applies small micro-interaction) and handle mobile nav
document.addEventListener("DOMContentLoaded", function () {
  // Hide loader if accidentally visible
  const loader = document.getElementById("loader");
  if (loader) loader.classList.add("hidden");

  // Add a tiny pop animation for any .result-card present
  const results = document.querySelectorAll(".result-card");
  results.forEach((r, idx) => {
    r.style.transformOrigin = "center top";
    r.style.opacity = "0";
    setTimeout(() => {
      r.classList.add("reveal");
    }, 120 + idx * 80);
  });

  // Add hover micro-lift for result card
  const cards = document.querySelectorAll(".result-card");
  cards.forEach(c => {
    c.addEventListener("mouseenter", () => c.style.transform = "translateY(-4px)");
    c.addEventListener("mouseleave", () => c.style.transform = "");
  });

  /* -------------------------------
     Mobile nav toggle (accessible)
     ------------------------------- */
  const navToggle = document.querySelector(".nav-toggle");
  const primaryMenu = document.getElementById("primary-menu");

  if (navToggle && primaryMenu) {
    navToggle.addEventListener("click", function (e) {
      const expanded = navToggle.getAttribute("aria-expanded") === "true";
      navToggle.setAttribute("aria-expanded", String(!expanded));
      primaryMenu.classList.toggle("open");
    });

    // Close mobile menu on resize to desktop
    window.addEventListener("resize", function () {
      if (window.innerWidth >= 1024) {
        primaryMenu.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
      }
    });

    // Close menu when clicking outside
    document.addEventListener("click", function (e) {
      if (!primaryMenu.contains(e.target) && !navToggle.contains(e.target) && primaryMenu.classList.contains("open")) {
        primaryMenu.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
      }
    });

    // Optional: close on Escape
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && primaryMenu.classList.contains("open")) {
        primaryMenu.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
        navToggle.focus();
      }
    });
  }
});
