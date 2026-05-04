// Copyright Epic Games, Inc. All Rights Reserved.

using System.IO;
using UnrealBuildTool;
using System.Collections.Generic;

public class BinaryGDAL : ModuleRules
{
    private string ModulePath
    {
        get { return ModuleDirectory; }
    }

    private string ThirdPartyPath
    {
        get { return Path.Combine(ModulePath, "ThirdParty"); }

    }

    public BinaryGDAL(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        //PublicDefinitions.Add("GEOS_DEBUG=0");
        bEnableUndefinedIdentifierWarnings = false;
        bUsePrecompiled = true;
        PublicIncludePaths.AddRange(
            new string[] {
            }
            );


        PrivateIncludePaths.AddRange(
            new string[] {
				// ... add other private include paths required here ...
			}
            );


        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "Projects"
				// ... add other public dependencies that you statically link with here ...
			}
            );


        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
				// ... add private dependencies that you statically link with here ...
				"Engine",
                "CoreUObject"
            }
            );


        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {
				// ... add any modules that your module loads dynamically here ...
			}
            );

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            // Include path
            string GDALIncludePath = Path.Combine(ThirdPartyPath, "include");
            PublicIncludePaths.Add(GDALIncludePath);

            // Libraries path
            string GDALibFolder = Path.Combine(ThirdPartyPath, "lib");
            // libs
            List<string> AllLibs = new List<string>();
            AllLibs.Add("zstd.lib");
            AllLibs.Add("bz2.lib");
            AllLibs.Add("charset.lib");
            AllLibs.Add("freexl.lib");
            AllLibs.Add("gdal.lib");
            AllLibs.Add("geos.lib");
            AllLibs.Add("geos_c.lib");
            AllLibs.Add("geotiff_i.lib");
            AllLibs.Add("gif.lib");
            AllLibs.Add("hdf5.lib");
            AllLibs.Add("hdf5_cpp.lib");
            AllLibs.Add("hdf5_hl.lib");
            AllLibs.Add("hdf5_hl_cpp.lib");
            AllLibs.Add("iconv.lib");
            AllLibs.Add("jpeg.lib");
            AllLibs.Add("json-c.lib");
            AllLibs.Add("json-c-static.lib");
            AllLibs.Add("kmlbase.lib");
            AllLibs.Add("kmlconvenience.lib");
            AllLibs.Add("kmldom.lib");
            AllLibs.Add("kmlengine.lib");
            AllLibs.Add("kmlregionator.lib");
            AllLibs.Add("kmlxsd.lib");
            AllLibs.Add("Lerc.lib");
            //AllLibs.Add("libcrypto.lib");
            AllLibs.Add("libcurl.lib");
            AllLibs.Add("libecpg.lib");
            AllLibs.Add("libecpg_compat.lib");
            AllLibs.Add("libexpat.lib");
            AllLibs.Add("libpgcommon.lib");
            AllLibs.Add("libpgport.lib");
            AllLibs.Add("libpgtypes.lib");
            //AllLibs.Add("libpng16.lib");
            AllLibs.Add("libpq.lib");
            AllLibs.Add("libsharpyuv.lib");
            //AllLibs.Add("libssl.lib");
            AllLibs.Add("libwebp.lib");
            AllLibs.Add("libwebpdecoder.lib");
            AllLibs.Add("libwebpdemux.lib");
            AllLibs.Add("libwebpmux.lib");
            AllLibs.Add("libxml2.lib");
            AllLibs.Add("lz4.lib");
            AllLibs.Add("lzma.lib");
            AllLibs.Add("minizip.lib");
            AllLibs.Add("netcdf.lib");
            AllLibs.Add("openjp2.lib");
            AllLibs.Add("pcre2-8.lib");
            AllLibs.Add("pcre2-16.lib");
            AllLibs.Add("pcre2-32.lib");
            AllLibs.Add("pcre2-posix.lib");
            AllLibs.Add("pkgconf.lib");
            AllLibs.Add("proj.lib");
            AllLibs.Add("qhull_r.lib");
            AllLibs.Add("qhullcpp.lib");
            AllLibs.Add("spatialite.lib");
            AllLibs.Add("sqlite3.lib");
            AllLibs.Add("szip.lib");
            AllLibs.Add("tiff.lib");
            AllLibs.Add("turbojpeg.lib");
            AllLibs.Add("uriparser.lib");
            AllLibs.Add("zlib.lib");

            foreach (string libName in AllLibs)
            {
                string LibPath = Path.Combine(GDALibFolder, libName);
                if (!File.Exists(LibPath))
                {
                    string Err = string.Format("Library '{0}' not found.", LibPath);
                    System.Console.WriteLine(Err);
                    throw new BuildException(Err);
                }

                PublicAdditionalLibraries.Add(LibPath);
            }

            // dlls
            List<string> AllDlls = new List<string>();
            AllDlls.Add("zstd.dll");
            AllDlls.Add("bz2.dll");
            AllDlls.Add("charset-1.dll");
            AllDlls.Add("freexl-1.dll");
            AllDlls.Add("gdal.dll");
            AllDlls.Add("geos.dll");
            AllDlls.Add("geos_c.dll");
            AllDlls.Add("geotiff.dll");
            AllDlls.Add("gif.dll");
            AllDlls.Add("hdf5.dll");
            AllDlls.Add("hdf5_cpp.dll");
            AllDlls.Add("hdf5_hl.dll");
            AllDlls.Add("hdf5_hl_cpp.dll");
            AllDlls.Add("iconv-2.dll");
            AllDlls.Add("jpeg62.dll");
            AllDlls.Add("json-c.dll");
            AllDlls.Add("legacy.dll");
            AllDlls.Add("Lerc.dll");
            AllDlls.Add("libcrypto-3-x64.dll");
            AllDlls.Add("libcurl.dll");
            AllDlls.Add("libecpg.dll");
            AllDlls.Add("libecpg_compat.dll");
            AllDlls.Add("libexpat.dll");
            AllDlls.Add("liblzma.dll");
            AllDlls.Add("libpgtypes.dll");
            AllDlls.Add("libpng16.dll");
            AllDlls.Add("libpq.dll");
            AllDlls.Add("libsharpyuv.dll");
            AllDlls.Add("libssl-3-x64.dll");
            AllDlls.Add("libwebp.dll");
            AllDlls.Add("libwebpdecoder.dll");
            AllDlls.Add("libwebpdemux.dll");
            AllDlls.Add("libwebpmux.dll");
            AllDlls.Add("libxml2.dll");
            AllDlls.Add("lz4.dll");
            AllDlls.Add("minizip.dll");
            AllDlls.Add("netcdf.dll");
            AllDlls.Add("openjp2.dll");
            AllDlls.Add("pcre2-8.dll");
            AllDlls.Add("pcre2-16.dll");
            AllDlls.Add("pcre2-32.dll");
            AllDlls.Add("pcre2-posix.dll");
            AllDlls.Add("pkgconf-3.dll");
            AllDlls.Add("proj.dll");
            AllDlls.Add("qhull_r.dll");
            AllDlls.Add("spatialite.dll");
            AllDlls.Add("sqlite3.dll");
            AllDlls.Add("szip.dll");
            AllDlls.Add("tiff.dll");
            AllDlls.Add("turbojpeg.dll");
            AllDlls.Add("uriparser.dll");
            AllDlls.Add("zlib1.dll");

            // Libraries path
            string GDALDllFolder = Path.Combine(ThirdPartyPath, "bin");
            string TempProjectBinaryDir = Path.Combine(ModulePath, "../../../../Binaries/Win64");
            if (!Directory.Exists(TempProjectBinaryDir))
            {
                Directory.CreateDirectory(TempProjectBinaryDir);
            }
            foreach (string dllName in AllDlls)
            {
                //PublicDelayLoadDLLs.Add(dll);
                string DllSrcPath = Path.Combine(GDALDllFolder, dllName);
                string DllDstPath = Path.Combine(ModulePath, "../../../../Binaries/Win64", Path.GetFileName(DllSrcPath));
                System.Console.WriteLine(DllDstPath);
                if (!File.Exists(DllDstPath) && File.Exists(DllSrcPath))
                {
                    File.Copy(DllSrcPath, DllDstPath);
                }

                //RuntimeDependencies.Add(DllDstPath);
                RuntimeDependencies.Add("$(BinaryOutputDir)/" + dllName, DllSrcPath);
            }

            RuntimeDependencies.Add("$(BinaryOutputDir)/proj/*", Path.Combine(ThirdPartyPath, "proj/*"), StagedFileType.SystemNonUFS);
            RuntimeDependencies.Add("$(BinaryOutputDir)/tools/*", Path.Combine(ThirdPartyPath, "tools/*"), StagedFileType.SystemNonUFS);
        }
    }
}
