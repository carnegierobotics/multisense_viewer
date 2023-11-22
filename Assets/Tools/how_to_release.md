## Create new git branch

1. Create new version main branch. Example = v1.2.0/main 
2. Change version number in top of CMakeLists file
3. (Optional) Update release description in workflow.yml or remember to do that before releasing draft
4. Add features to this new version in separate branches if required, such as v1.2.0/feature/record_imu. 
Otherwise just make changes to v1.2.0/main
5. Once finished merge v1.2.0/main to master by completing the steps below

## Create new release
1. Create git tag, Must be same as in cmake format: v*.*.*
2. Create pull request and await approval.
 - Check that the github runner can succesfully compile the project.
    reissue workflow command by using gh: gh workflow run workflow_pr.yml -r {branch-name: i.e. v1.1.0/main}
3. Merge with master
4. A runner will pick up and do the following:
 - Create a draft release
 - Compile Ubuntu version
 - Compile Windows version
 - Create Ubuntu/Windows installers
 - Upload to draft release

 It should succeed if worfklow_pr.yml has succeeded as well

6. A user must then edit and publish the release manually. Remember signing the windows .exe files