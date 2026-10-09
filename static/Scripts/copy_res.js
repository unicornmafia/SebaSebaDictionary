// Copy-to-clipboard button for RES text in the Gardiner Signs column.
// Buttons may be added dynamically (faulkner.js), so clicks are delegated.
(function () {
    function fallbackCopy(text) {
        // navigator.clipboard needs a secure context; plain-http LAN access doesn't have one
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        ta.setSelectionRange(0, text.length);
        var ok = false;
        try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
        document.body.removeChild(ta);
        return ok ? Promise.resolve() : Promise.reject();
    }

    function copyText(text) {
        if (navigator.clipboard && window.isSecureContext) {
            return navigator.clipboard.writeText(text).catch(function () { return fallbackCopy(text); });
        }
        return fallbackCopy(text);
    }

    document.addEventListener("click", function (e) {
        var btn = e.target.closest(".copy-res");
        if (!btn) return;
        e.preventDefault();
        var text = btn.parentElement.querySelector(".res-text").textContent.trim();
        var icon = btn.querySelector("i");
        copyText(text).then(function () {
            icon.className = "fas fa-check";
            btn.classList.add("copied");
        }, function () {
            icon.className = "fas fa-times";
        }).then(function () {
            setTimeout(function () {
                icon.className = "far fa-copy";
                btn.classList.remove("copied");
            }, 1200);
        });
    });
})();
