function showTabContent(tabId) {
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.remove('active-content'));
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => tab.classList.remove('active-tab'));
    document.getElementById(tabId).classList.add('active-content');
    const activeTab = document.querySelector('.tab[onclick="showTabContent(\'' + tabId + '\')"]');
    activeTab.classList.add('active-tab');
}

function toggleShowMoreContent(button) {
    const content = button.nextElementSibling;
    if(content.style.display === "none") content.style.display = "block";
    else content.style.display = "none";
}