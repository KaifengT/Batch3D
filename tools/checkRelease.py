import urllib.request
import json


def checkReleaseonGithub(owner, repo, current_version):
    '''
    Check for updates from a GitHub repository.
    Args:
        owner (str): GitHub repository owner.
        repo (str): GitHub repository name.
        current_version (str): Current version string, e.g., "v1.0.0".
    Returns:
        dict: A dictionary containing update information.
    '''

    try:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        
        request = urllib.request.Request(
            api_url,
            headers={
                'User-Agent': 'Batch3D',
                'Accept': 'application/vnd.github.v3+json'
            }
        )
        
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        for release in data:
            if not release.get('prerelease', False):
                latest_version = release['tag_name']
                
                latest_nums = [int(x) for x in latest_version.lstrip('v').split('.')]
                current_nums = [int(x) for x in current_version.lstrip('v').split('.')]
                
                has_update = False
                for i in range(min(len(latest_nums), len(current_nums))):
                    if latest_nums[i] > current_nums[i]:
                        has_update = True
                        break
                    elif latest_nums[i] < current_nums[i]:
                        break
                
                if not has_update and len(latest_nums) > len(current_nums):
                    has_update = True
                
                return {
                    "has_update": has_update,
                    "current_version": current_version,
                    "latest_version": latest_version,
                    "download_url": release['html_url'] if has_update else None
                }
        
        return {"has_update": False, "error": "No releases found"}
        
    except Exception as e:
        return {"has_update": False, "error": f"Check failed: {str(e)}"}

