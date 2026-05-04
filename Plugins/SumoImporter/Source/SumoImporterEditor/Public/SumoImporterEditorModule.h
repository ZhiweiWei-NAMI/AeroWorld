#pragma once

#include "Modules/ModuleManager.h"

class FSumoImporterEditorModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	void RegisterMenus();
	void HandleImportNetXmlClicked();
	void ShowResultDialog(const FString& Title, const FString& Message, bool bIsError) const;
};
